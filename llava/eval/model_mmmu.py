import torch
import os, sys
import random

import numpy as np
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init

from argparse import ArgumentParser

from llava.eval.mmmu_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG
from llava.eval.mmmu_utils import call_llava_engine_df, llava_image_processor
from llava.eval.mmmu_utils import parse_multi_choice_response, parse_open_response

config = {
    'task_instructions': "",
    'multi_choice_example_format': "{}\n\n{}\n\nAnswer with the option's letter from the given choices directly.",
    'short_ans_example_format': "{}\n\nAnswer the question using a single word or phrase."
}


def run_model(args, samples, model, call_model_engine_fn=None, tokenizer=None, processor=None):
    out_samples = dict()
    with torch.no_grad():
        for sample in tqdm(samples):
            response = call_model_engine_fn(args, sample, model, tokenizer, processor)

            if sample['question_type'] == 'multiple-choice':
                pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
            else:  # open question
                pred_ans = response
            out_samples[sample['id']] = pred_ans
    return out_samples

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)


    # print('llava_initializing...')
    processor = None
    call_model_engine = call_llava_engine_df
    vis_process_func = llava_image_processor

    # # load config and process to one value
    # args.config = load_yaml(args.config_path)
    # for key, value in args.config.items():
    #     if key != 'eval_params' and type(value) == list:
    #         assert len(value) == 1, 'key {} has more than one value'.format(key)
    #         args.config[key] = value[0]

    # run for each subject
    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values():
        sub_dataset = load_dataset(args.data_path, subject, split=args.split)
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)


    # load model
    # model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, vis_processors, _ = load_pretrained_model(args.model_path, None,
    #                                                             model_name)

    samples = []
    for sample in dataset:

        sample = process_single_sample(sample)
        sample = construct_prompt(sample, config)
        if sample['image']:
            sample['image'] = vis_process_func(sample['image'], image_processor).to(device)
        samples.append(sample)

    # run ex
    out_samples = run_model(args, samples, model, call_model_engine, tokenizer, processor)

    save_json(args.output_path, out_samples)
    # metric_dict.update({"num_example": len(out_samples)})
    # save_json(save_result_path, metric_dict)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='llava1.5_13b_val.json',
                        help='name of saved json')
    parser.add_argument('--config_path', type=str, default="configs/llava1.5.yaml")
    parser.add_argument('--data_path', type=str, default="MMMU/MMMU") # hf dataset path.
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    main(args)
