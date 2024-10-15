import argparse
import torch
import os, sys
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.grad_utils import collect_grads

from PIL import Image
import math

import torch.nn as nn
log_softmax = nn.LogSoftmax(dim=-1)
nll_loss = nn.NLLLoss(reduction='none')
loss_fct = nn.CrossEntropyLoss()


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: v for k, v in to_return.items()}
    return to_return


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def compute_loss(logits, labels, vocab_size):

    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


# Used to get the ppl and emb for the whole input
def get_perplexity_and_embedding_whole_text(model, input_ids, images, image_sizes, tokenizer, args, embedding=False):

    if args.grads:

        assert args.lora, "Full model gradients are too high-dimensional for scalable storage"

        outputs = model(input_ids, images=images, image_sizes=image_sizes, labels=input_ids.contiguous(), output_hidden_states=True)
        # state_dict = get_peft_state_maybe_zero_3(
        #         model.named_parameters(), "none"
        #     )
        # print(state_dict)
        # for k, v in state_dict.items():
        #     print(k, v.shape)
        # for name, param in model.named_parameters():
            # print(name)
            # if name == 'model.layers.31.self_attn.q_proj.weight':
            #     print(param[0])

        # grads = model.model.mm_projector

    else:
        with torch.no_grad(): 
            outputs = model(input_ids, images=images, image_sizes=image_sizes, labels=input_ids.contiguous(), output_hidden_states=embedding)


    logits = outputs.logits
    loss = outputs.loss
    # loss = compute_loss(logits, input_ids.contiguous(), model.config.vocab_size)
    # print(loss, outputs.loss)
    perplexity = torch.exp(loss)

    # with torch.inference_mode():
    #     output_ids = model.generate(
    #         input_ids,
    #         images=images,
    #         image_sizes=image_sizes,
    #         do_sample=True if args.temperature > 0 else False,
    #         temperature=args.temperature,
    #         max_new_tokens=1024,
    #         use_cache=True,
    #     )
    # loss = outputs.loss
    # perplexity = torch.exp(loss)

    # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # print(outputs)

    if embedding:
        hidden_states = outputs.hidden_states
        sentence_embeddings = []
        for layer_dim in args.layer_dims.split(","):
            layer_dim = int(layer_dim)
            sentence_embeddings.append(hidden_states[layer_dim].mean(dim=1).to("cpu"))
        
        # embeddings_0 = hidden_states[0]
        # sentence_embedding_0 = embeddings_0.mean(dim=1)
        # embeddings = hidden_states[-1]
        # sentence_embedding = embeddings.mean(dim=1)

        # return perplexity.to('cpu').item(), sentence_embedding_0.to("cpu"), sentence_embedding.to('cpu')
        return perplexity.to('cpu').item(), sentence_embeddings
    
    else:
        return perplexity.to('cpu').item(), None


# Used to get the ppl and emb for the whole input
def get_image_embedding(model, input_ids, images, image_sizes):

    with torch.no_grad(): 
        (_, _, _, _, inputs_embeds, _,) = model.prepare_inputs_labels_for_multimodal(input_ids, None, None, None, 
                                                                                  labels=input_ids.contiguous(), images=images, 
                                                                                  image_sizes=image_sizes, 
                                                                                  )
    print(inputs_embeds.shape)
    return inputs_embeds.mean(dim=1).squeeze().detach().cpu()

# Used to get the ppl and emb for part of input, used in conditional version, and token-wise loss
def get_perplexity_and_embedding_part_text(tokenizer, model, text, target_span, max_length):

    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    start_index = text.rfind(target_span)
    start_token = len(tokenizer.encode(text[:start_index]))
    end_token = input_ids.shape[1]

    labels = input_ids.clone()
    labels[0, :start_token] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    loss = outputs.loss
    perplexity = torch.exp(loss)

    losses = []
    logits = outputs.logits
    for i in range(1, end_token):
        log_prob_dist = log_softmax(logits[0, i-1])
        true_token = input_ids[0, i]
        token_loss = nll_loss(log_prob_dist.unsqueeze(0), true_token.unsqueeze(0))
        losses.append(token_loss.item())

    return perplexity.to('cpu'), 0, losses


def eval_model(args):

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, merge_lora=not args.grads)

    # sys.exit()

    # for name, param in model.named_parameters():
    #     # if name == 'model.layers.31.self_attn.q_proj.weight':
    #     print(name, param.shape)

    # sys.exit()

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    out_data = []
    embeddings = [[] for i in range(len(args.layer_dims.split(',')))]
    image_embeddings = []
    skip = 0

    for i, line in enumerate(tqdm(questions)):


        try:

            idx = line["id"]
            temp_data_i = {"id": idx}
            question = line['conversations'][0]
            qs = question['value'].replace('<image>', '').strip()
            cur_prompt = qs

            if 'image' in line:

                # get likelihood with image
                image_file = line["image"]
                image = Image.open(os.path.join(args.image_folder, image_file))
                image_tensor = process_images([image], image_processor, model.config)[0]
                images = image_tensor.unsqueeze(0).half().cuda()
                image_sizes = [image.size]
                if getattr(model.config, 'mm_use_im_start_end', False):
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                cur_prompt = '<image>' + '\n' + cur_prompt
                if args.single_pred_prompt:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
                    cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

                conv = conv_templates[args.conv_mode].copy()
                roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

                line['conversations'][0]['value'] = line['conversations'][0]['value'].replace('<image>', '').strip()
                source = line["conversations"]
                if roles[source[0]["from"]] != conv.roles[0]:
                    # Skip the first one if it is not from human
                    source = source[1:]

                conv.messages = []
                for j, sentence in enumerate(source):
                    role = roles[sentence["from"]]
                    assert role == conv.roles[j % 2], f"{i}"
                    conv.append_message(role, sentence["value"])

                prompt = conv.get_prompt() # Fix conversation input
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                # print("############ With Image")
                # ppl_with_image, sent_embed_0, sent_embed = get_perplexity_and_embedding_whole_text(model, input_ids, images, image_sizes, tokenizer, args, embedding=args.embedding)
                print('Sample with image')
                ppl_with_image, sent_embeds = get_perplexity_and_embedding_whole_text(model, input_ids, images, image_sizes, tokenizer, args, embedding=args.embedding)
                if args.image_embeds:
                    image_embeds = get_image_embedding(model, input_ids, images, image_sizes)
                    image_embeddings.append(image_embeds)
                #####################################################################

            else:
                images = None
                image_sizes = None
                ppl_with_image = -1
                sent_embeds = None


            idx = line["id"]
            question = line['conversations'][0]
            qs = question['value'].replace('<image>', '').strip()
            cur_prompt = qs

            # get likelihood without image
            images = None
            image_sizes = None

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            # conv.append_message(conv.roles[1], line['conversations'][1]['value'])
            conv.append_message(conv.roles[1], None)

            prompt = conv.get_prompt()
            # print(prompt)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            # print("############ Without Image")
            # ppl_wo_image, sent_embed_no_image_0, sent_embed_no_image = get_perplexity_and_embedding_whole_text(model, input_ids, None, None, tokenizer, args, embedding=args.embedding or sent_embed is None)
            ppl_wo_image, sent_embeds_no_image = get_perplexity_and_embedding_whole_text(model, input_ids, None, None, tokenizer, args, embedding=args.embedding and sent_embeds is None)

            if sent_embeds is not None:
                for j, emb in enumerate(sent_embeds):
                    embeddings[j].append(emb)
            else:
                assert sent_embeds_no_image is not None
                for j, emb in enumerate(sent_embeds_no_image):
                    embeddings[j].append(emb)

            temp_data_i['ppl'] = [ppl_with_image, ppl_wo_image]
            # print(temp_data_i)
            out_data.append(temp_data_i)

            print("Logged ppl %.4f with image and %.4f without image for sample id %s" % (ppl_with_image, ppl_wo_image, idx))

        except Exception as e:
            print("Exception: ", e)
            print("Skipping: ")
            print(line)
            skip += 1
            continue

        # if (i+1) % 10 == 0:
        #     break

    with open(args.save_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print("Skipped %s samples" % skip)

    # for layer_dim, embs in zip(args.layer_dims.split(','), embeddings):
    #     if layer_dim=="-1":
    #         torch.save(torch.cat(embs, dim=0), args.save_path.replace('.json', '.pt'))
    #     else:
    #         torch.save(torch.cat(embs, dim=0), args.save_path.replace('.json', '_dim=%s.pt' % layer_dim))

    if args.image_embeds:
        torch.save(torch.stack(image_embeddings), args.save_path.replace('.json', '-image-embeds.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--save-path", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--grads", action="store_true")
    parser.add_argument("--image-embeds", action="store_true")
    parser.add_argument("--layer-dims", type=str, default="-1") # default is the final layer before classifier
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--embedding", action="store_true")
    args = parser.parse_args()

    eval_model(args)
