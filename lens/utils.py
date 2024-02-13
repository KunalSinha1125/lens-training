import datetime
import os

import torch
from torch.distributed import init_process_group
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from PIL import Image
import requests
import tqdm

default_device = "cuda" if torch.cuda.is_available() else "cpu"

MAP_CLIP_NAME = {
    "openai/clip-vit-large-patch14": "ViT-L-14",
    "openai/clip-vit-base-patch16": "ViT-B-16",
    "openai/clip-vit-base-patch32": "ViT-B-32",
    "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K": "laion-ViT-H-14-2B",
    "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k": "laion-ViT-bigG-14-2B",
}


def ddp_setup():
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=180000))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def create_sampler(dataset, distributed=False):
    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    return sampler


def create_dataloader(dataset, batch_size=8, num_workers=0, dataset_name="cifar10"):
    def collate_fn(data):
        if dataset_name == "RIW/small-coco":
            batch = {'image': [], 'question': [], 'answer': []}
            for d in data:
                batch['image'] = Image.open(requests.get(d['image_id'], stream=True).raw).convert('RGB')
                batch['question'] = d['question']
                answer_id = np.argmax(d['label']['weights'])
                batch['answer'] = d['label']['ids'][answer_id]
            return batch
        elif dataset_name == "cifar10":
            classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
            return {
                'image': [d['img'] for d in data],
                'caption': [classes[int(d['label'])] for d in data]
            }
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn
    )
    return loader


def is_main_process():
    if int(os.environ["RANK"]) == 0:
        return True
    else:
        return False


def get_llm_model(version, load_8bit, device_map=None):
    if load_8bit:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                version,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map={"": device_map},
            )
        except:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                version,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map={"": device_map},
            )
    else:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(version).to(device_map)
        except:
            model = AutoModelForCausalLM.from_pretrained(version).to(device_map)
    model = model.eval()
    return model


def create_prompt_sample(
    samples,
    idx,
    desc_idx=0,
    tags_col="tags",
    attributes_col="attributes",
    caption_col="caption",
    intensive_captions_col="intensive_captions",
    question_col="questions",
    question_prompt=None,
    num_intensive_captions=5,
    mode="all",
):
    prompt = ""
    if question_prompt is not None:
        question = question_prompt
    else:
        question = samples[question_col][idx]

    if mode == "vqa":
        prompt += "Image:\n"
        prompt += "Captions:"
        prompt += ".".join(
            samples[intensive_captions_col][idx][:num_intensive_captions]
        )
        prompt += "\nQuestion:"
        prompt += question
        prompt += "\nShort Answer:"

    elif mode == "vision":
        prompt += "Tag: "
        prompt += ",".join(samples[tags_col][idx])
        prompt += "\nAttributes: "
        prompt += ",".join(samples[attributes_col][idx])
        prompt += "\nQuestion:"
        prompt += question
        prompt += "\nShort Answer:"

    elif mode == "tags_only":
        prompt += "Tag: "
        prompt += ",".join(samples[tags_col][idx])
        prompt += "\nQuestion:"
        prompt += question
        prompt += "\nShort Answer:"

    elif mode == "tags_only_single":
        prompt += "\nTag: "
        prompt += samples[tags_col][idx][desc_idx]
        prompt += "\nQuestion:"
        prompt += question
        prompt += "\nShort Answer:"

    elif mode == "tags_only_loop":
        tags = samples[tags_col][idx]
        prompt += "\nTag: "
        prompt += ",".join(tags[:desc_idx] + tags[desc_idx+1:])
        prompt += "\nQuestion: "
        prompt += question
        prompt += "\nShort Answer: "

    elif mode == "tags_only_single_phi2":
        tag = samples[tags_col][idx][desc_idx]
        #prompt = f"Instruct: Describe the image whose tag is {tag}.\nOutput:"
        #prompt = f'''Alice: can you tell me what you see in this image? The image tags are [{tag}].\nBob:'''
        prompt = f"Describe the given image."

    elif mode == "attributes_only":
        prompt += "Attribute: "
        prompt += ",".join(samples[attributes_col][idx])
        prompt += "\nQuestion:"
        prompt += question
        prompt += "\nShort Answer:"

    elif mode == "attributes_only_single":
        prompt += "Attribute: "
        prompt += samples[attributes_col][idx][desc_idx]
        prompt += "\nQuestion:"
        prompt += question
        prompt += "\nShort Answer:"
    
    elif mode == "attributes_only_loop":
        attributes = samples[attributes_col][idx]
        prompt += "Attribute: "
        prompt += ",".join(attributes[:desc_idx] + attributes[desc_idx+1:])
        prompt += "\nQuestion: "
        prompt += question
        prompt += "\nShort Answer: "

    elif mode == "intensive_captions_only_single":
        prompt += "Caption: "
        prompt += samples[intensive_captions_col][idx][desc_idx]
        prompt += "\nQuestion:"
        prompt += question
        prompt += "\nShort Answer:"

    elif mode == "hm":
        prompt += "Image:\n"
        prompt += "Caption:"
        prompt += samples[caption_col][idx]
        prompt += "\nAttributes:"
        prompt += ",".join(samples[attributes_col][idx])
        prompt += "\nTags:"
        prompt += ",".join(samples[attributes_col][idx])
        prompt += "\nQuestion: Is the image hateful or not-hateful?"
        prompt += "\nShort Answer:"

    elif mode == "all":
        prompt += "Tags:\n-"
        prompt += "\n-".join(samples[tags_col][idx])
        prompt += "\nAttributes:\n-"
        prompt += "\n-".join(samples[attributes_col][idx])
        prompt += "\nQuestion:"
        prompt += question
        prompt += "\nShort Answer:"
    else:
        raise Exception("Mode not available")
    return prompt


