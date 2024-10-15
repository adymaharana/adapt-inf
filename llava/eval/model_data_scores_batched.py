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
from llava.grad_utils import collect_grads, collect_grads_and_reps, collect_reps
from llava.train.train import make_supervised_data_module
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
import transformers

from PIL import Image
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch.nn as nn
log_softmax = nn.LogSoftmax(dim=-1)
nll_loss = nn.NLLLoss(reduction='none')
loss_fct = nn.CrossEntropyLoss()



@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="facebook/opt-125m")
    model_base: Optional[str] = field(default="")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")

@dataclass
class EvalArguments:
    output_dir: Optional[str] = field(default="")
    conv_mode: Optional[str] = field(default="")
    num_chunks: Optional[int] = field(default=1)
    chunk_idx: Optional[int] = field(default=0)
    grads: bool = False
    reps: bool = False
    layer_dims: list[int] = field(default_factory= lambda: [31])
    single_pred_prompt: bool = False
    eval_batch_size: int = field(default=1)
    dataloader_num_workers: int = field(default=1)
    dataloader_prefetch_factor: int = field(default=1)
    max_samples: Optional[int] = field(default=None)
    # metrics: list[str] = field(default_factory= lambda: ['ppl'])
    gradient_type: Optional[str] = field(default='sgd')
    metrics: Optional[list[str]] = field(default=None)
    zero_order: bool = False



@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


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


def eval_model(model_args, data_args, eval_args):

    print(model_args)
    print(eval_args)
    print(data_args)

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(model_args.model_path)
    model_name = get_model_name_from_path(model_path)
    data_args.mm_use_im_start_end = model_args.mm_use_im_start_end # It's all set to False in default finetuning scripts
    # model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_args.model_base, model_name, 
                                                                           merge_lora=False if eval_args.grads else True, 
                                                                           make_trainable=True if eval_args.grads else False)

    data_args.image_processor = image_processor
    data_args.is_multimodal = True

    # chunked data
    questions = json.load(open(os.path.expanduser(data_args.data_path), "r"))
    questions = get_chunk(questions, eval_args.num_chunks, eval_args.chunk_idx)
    data_file = os.path.split(data_args.data_path)[-1]
    chunked_data_path = os.path.join(eval_args.output_dir, data_file.replace('.json', '_%s.json' % eval_args.chunk_idx))
    os.makedirs(eval_args.output_dir, exist_ok=True)
    with open(chunked_data_path, 'w') as f:
        json.dump(questions, f, indent=2)
        print(f"Saved {len(questions)} questions to {chunked_data_path}")
    data_args.data_path = chunked_data_path

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)


    dataloader_params = {
        "batch_size": eval_args.eval_batch_size,
        "collate_fn": data_module['data_collator'],
        "num_workers": eval_args.dataloader_num_workers,
        "pin_memory": True,
        "persistent_workers": False
    }

    eval_dataset = data_module['train_dataset']
    if not isinstance(eval_dataset, IterableDataset):
        dataloader_params["sampler"] = SequentialSampler(eval_dataset)
        dataloader_params["drop_last"] = False
        dataloader_params["prefetch_factor"] = eval_args.dataloader_prefetch_factor

    # accelerator.free_memory() will destroy the references, so
    # we need to store the non-prepared version
    eval_dataloader = DataLoader(eval_dataset, **dataloader_params)

    if eval_args.grads and eval_args.reps:

        if eval_args.gradient_type == 'adam':
            optimizer_path = os.path.join(model_args.model_path, "optimizer_state.pt")
        else:
            optimizer_path = None
        
        # TODO: Fix the Adam optimizer state loading? And saving?
        collect_grads_and_reps(eval_dataloader, model, eval_args.output_dir, 
                    layer_dim=eval_args.layer_dims, 
                    gradient_type=eval_args.gradient_type,
                    adam_optimizer_state=None if optimizer_path is None else torch.load(optimizer_path, map_location="cpu")["state"], 
                    max_samples=eval_args.max_samples,
                    metrics=eval_args.metrics)
    
    elif eval_args.grads:

        if eval_args.gradient_type == 'adam':
            optimizer_path = os.path.join(model_args.model_path, "optimizer_state.pt")
        else:
            optimizer_path = None
        
        # TODO: Fix the Adam optimizer state loading? And saving?
        collect_grads(eval_dataloader, model, eval_args.output_dir, 
                    layer_dim=eval_args.layer_dims, 
                    gradient_type=eval_args.gradient_type,
                    adam_optimizer_state=None if optimizer_path is None else torch.load(optimizer_path, map_location="cpu")["state"], 
                    max_samples=eval_args.max_samples,
                    metrics=eval_args.metrics,
                    zero_order=eval_args.zero_order)
    
    elif eval_args.reps:
        collect_reps(eval_dataloader, model, eval_args.output_dir, 
                    max_samples=eval_args.max_samples)

    else:
        raise NotImplementedError


parser = transformers.HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

eval_model(model_args, data_args, eval_args)
