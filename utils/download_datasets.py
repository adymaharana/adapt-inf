import os
from tqdm import tqdm
from config import BASE_IMG_DIR, LLAVA_RAW_PATH, LLAVA_PATH, LLAVA_IMG_DIR, M3IT_IMG_DIR, M3IT_PATH, MINIGPT4_IMG_DIR, MINIGPT4_RAW_PATH, MINIGPT4_PATH, MANTIS_PATH, MANTIS_IMG_DIR
import json
import io, base64
import argparse
from PIL import Image
import random
from datasets import load_dataset


m3it_dset2task = {
    "coco": "captioning",
    "textcap": "captioning",
    "image-paragraph-captioning": "captioning",
    "coco-goi": "classification",
    "coco-text": "classification",
    "imagenet": "classification",
    "coco-itm": "classification",
    "snli-ve": "classification",
    "mocheg": "classification",
    "iqa": "classification",
    "vqa-v2": "vqa",        
    "shapes": "vqa",
    "docvqa": "vqa",        
    "ocr-vqa": "vqa",        
    "st-vqa": "vqa",        
    "text-vqa": "vqa",        
    "gqa": "vqa",
    "okvqa": "kvqa",
    "a-okvqa": "kvqa",
    "science-qa": "kvqa",
    "viquae": "kvqa",
    "clevr": "reasoning",
    "nlvr": "reasoning",
    "vcr": "reasoning",
    "visual-mrc": "reasoning",
    "winoground": "reasoning",
    "vist": "generation",
    "visual-dialog": "generation",
    "multi30k": "generation",
    "fm-iqa": "chinese",
    "coco-cn": "chinese",
    "flickr8k-cn": "chinese",
    "chinese-food": "chinese",
    "mmchat": "chinese",
    "ss": "video",
    "ivqa": "video",
    "msvd-qa": "video",
    "activitynet-qa": "video",
    "msrvtt": "video",
    "msrvtt-qa": "video",
}

def parse_args():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--dataset', help='Select the dataset that needs to be downloaded or processed', required=True, type=str)
    # Parse the arguments
    args = parser.parse_args()
    return args


def preprocess_llava_dataset():

    data = json.load(open(LLAVA_RAW_PATH))

    include_idxs = []
    for i, d in tqdm(enumerate(data)):
        if "image" not in d:
            include_idxs.append(i)
            continue
        if os.path.exists(os.path.join(LLAVA_IMG_DIR, d["image"])):
            include_idxs.append(i)
        else:
            continue
    
    print("Saving %s samples out of %s" % (len(include_idxs), len(data)))
    with open(LLAVA_PATH, 'w') as f:
        json.dump([data[idx] for idx in include_idxs], f, indent=2)


def download_m3it_dataset(mode='train'):

    dsets = list(m3it_dset2task.keys())

    for ds_name in dsets:

        try:
            dataset = load_dataset("MMInstruction/M3IT", ds_name)
            dataset = dataset[mode]
        except:
            print("Skipping %s because of exception" % ds_name)
            continue

        if len(dataset) == 0:
            print("Skipping %s because no samples found in data set" % ds_name)
            continue
        
        new_samples = []
        img_save_dir = M3IT_IMG_DIR
        
        count = 0
        for idx, data_instance in tqdm(enumerate(dataset)):
            try:

                instruction = data_instance["instruction"]  # str
                inputs = data_instance["inputs"]  # str
                outputs = data_instance["outputs"]  # str

                if len(data_instance['image_base64_str']) == 1:
                    
                    save_name = ds_name + '_{}_'.format(mode) + str(idx) + '.jpg'
                    if not os.path.exists(os.path.join(img_save_dir, save_name)):
                        img = Image.open(io.BytesIO(base64.decodebytes(bytes(data_instance['image_base64_str'][0], "utf-8"))))
                        img.save(os.path.join(img_save_dir, save_name))
                    img_file = 'images/' + save_name
                
                else:
                    
                    img_file = []
                    for j in range(len(data_instance['image_base64_str'])):
                        
                        save_name = ds_name + '_{}_'.format(mode) + str(idx) + '_' + str(j) + '.jpg'
                        if not os.path.exists(os.path.join(img_save_dir, save_name)):
                            img = Image.open(io.BytesIO(base64.decodebytes(bytes(data_instance['image_base64_str'][j], "utf-8"))))
                            img.save(os.path.join(img_save_dir, save_name))
                        img_file.append('images/' + save_name)

                sample = {}
                sample['id'] = ds_name + '_{}_'.format(mode) + str(idx).zfill(12)
                sample['task'] = m3it_dset2task[ds_name]
                sample['image'] = img_file
                sample["conversations"] = []
                sample["conversations"].append({
                    "from": "human",
                    "value": "<image>\n" + instruction + ' ' + inputs
                })
                sample["conversations"].append({
                    "from": "gpt",
                    "value": outputs
                })
                new_samples.append(sample)
                count += 1
            except:
                continue

        print("Saved %s samples for %s dataset" % (count, ds_name))
        
        with open(M3IT_PATH, "w") as f:
            json.dump(new_samples, f, indent=2)


def preprocess_minigpt4_dataset():

    minigpt_data = json.load(open(MINIGPT4_RAW_PATH, "r"))["annotations"]
    prompts = ["Describe this image in detail.", 
               "Take a look at this image and describe what you notice."
               "Please provide a detailed description of the picture.",
               "Could you describe the contents of this image for me?"]
    
    new_samples = []
    for i, d in tqdm(enumerate(minigpt_data)):
        sample = {"id": str(i).zfill(12)}
        assert os.path.exists(os.path.join(MINIGPT4_IMG_DIR, d["image_id"] + '.jpg'))
        sample["image"] = 'image/' + d["image_id"] + '.jpg'
        sample["conversations"] = []
        sample["conversations"].append({
            "from": "human",
            "value": "<image>\n" + random.choice(prompts)
        })
        sample["conversations"].append({
            "from": "gpt",
            "value": d["caption"]
        })
        new_samples.append(sample)
    
    with open(MINIGPT4_PATH, "w") as f:
        json.dump(new_samples, f, indent=2)


def download_mantis_dataset():

    role_map = {'user': 'human', 'assistant': 'gpt'}
    
    img_save_dir = '/nas-hdd/tarbucket/adyasha/datasets/llava/mantis/images'
    os.makedirs(img_save_dir, exist_ok=True)

    dsets = ['docvqa', 'dvqa', 'chartqa', 'iconqa', 'lrv_multi', 'spot-the-diff']
    img_dir_suffix = '/nas-hdd/tarbucket/adyasha/cache/TIGER-Lab___mantis-instruct/'

    all_samples = []
    count = 0
    multi_image_count = 0
    for ds_name in dsets:
        try:
            dataset = load_dataset("TIGER-Lab/Mantis-Instruct", ds_name, revision="script")
            train_set = dataset["train"]
            print(ds_name, len(train_set))
        except Exception as e:
            print("Skipping %s because of exception" % ds_name, e)
            continue

        if len(train_set) == 0:
            print("Skipping %s because no samples found in train set" % ds_name)
            continue

        for idx in tqdm(len(dataset)):

            try:
                sample = train_set[idx].copy()
                if len(sample['images']) > 1:
                    multi_image_count += 1
                sample['image'] = [img['path'][len(img_dir_suffix):] for img in sample['images']]
                del sample['images']
                sample['conversations'] = [conv for conv in sample['conversation']]
                del sample['conversation']
                sample['conversations'] = [{'from': role_map[conv['role']], 'value': conv['content']} for conv in sample['conversations']]
                
            except Exception as e:
                print('Faced error', e)
                continue
            count += 1
            all_samples.append(sample.copy())
    
    random.shuffle(all_samples)
    print(f"Saving {count} samples ({multi_image_count} with multiple images) to dataset /nas-hdd/tarbucket/adyasha/datasets/llava/mantis/mantis_mmmu.json")
    
    with open(MANTIS_PATH, "w") as f:
        json.dump(all_samples, f, indent=2)


def main(args):

    if args.dataset == 'llava':
        preprocess_llava_dataset()
    elif args.dataset == 'm3it':
        download_m3it_dataset()
    elif args.dataset == 'minigpt':
        preprocess_minigpt4_dataset()
    elif args.dataset == 'mantis':
        download_mantis_dataset()
    else:
        raise NotImplementedError


if __name__ == "__main__":

    args = parse_args()
    main(args)