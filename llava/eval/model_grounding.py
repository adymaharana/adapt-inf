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
from datasets import load_dataset

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    ds = load_dataset("jxu124/refcoco-benchmark")['refcoco_unc_testA']
    answers_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    answers = []

    for i, line in enumerate(tqdm(ds)):
        
        question = "Please provide the bounding box coordinate of the region this sentence describes: {}."
        cur_prompt = question
        qs = question

        if 'image' in line:
            image = line["image"].convert('RGB')
            # image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)[0]
            images = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [image.size]
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None
            image_sizes = None

        sample = {"id": line["image_info"]["id"], "prompts": [], "outputs": []}

        for j, ann in enumerate(line['ref_list']):

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs.format(ann['ref_info']['sentences'][0]['sent']))
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            print(prompt)

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=32,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(outputs)

            sample['prompts'].append(prompt)
            sample['outputs'].append(outputs)

        answers.append(sample.copy())

    with open(answers_file, "w") as ans_file:
        json.dump(answers, ans_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="coco-cn")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--data-file", type=str, default="tables/question.json")
    parser.add_argument("--output-file", type=str, default="tables/question.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    args = parser.parse_args()

    eval_model(args)
