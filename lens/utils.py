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


def create_dataloader(dataset, batch_size=8, num_workers=0):
    def collate_fn(data):
        #batch = {'image': [], 'question': [], 'answer': []}
        #for d in data:
        #    batch['image'] = Image.open(requests.get(d['image_id'], stream=True).raw).convert('RGB')
        #    batch['question'] = d['question']
        #    answer_id = np.argmax(d['label']['weights'])
        #    batch['answer'] = d['label']['ids'][answer_id]
        #return batch
        return {
            'image': [d['image'] for d in data],
            'caption': [d['caption'] for d in data]
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
    num_intensive_captions=50,
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
        prompt += "Tag: "
        prompt += samples[tags_col][idx][desc_idx]
        prompt += "\nQuestion:"
        prompt += question
        prompt += "\nShort Answer:"

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

def compute_perplexity(model, prompt_encodings, label_encodings):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encodings = torch.cat([prompt_encodings, label_encodings], dim=-1)
    answer_len = label_encodings.input_ids.size(1)
    input_ids = encodings.input_ids.to(device)
    label_ids = input_ids.clone()
    label_ids[:, :-answer_len] = -100
    outputs = model(input_ids, labels=label_ids)
    return outputs.loss

    # nlls = []
    # prev_end_loc = 0
    # for begin_loc in tqdm(range(0, seq_len, stride)):
    #     end_loc = min(begin_loc + max_length, seq_len)
    #     trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    #     input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    #     target_ids = input_ids.clone()
    #     target_ids[:, :-answer_len] = -100

    #     with torch.no_grad():
    #         outputs = model(input_ids, labels=target_ids)

    #         # loss is calculated using CrossEntropyLoss which averages over valid labels
    #         # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
    #         # to the left by 1.
    #         neg_log_likelihood = outputs.loss

    #     nlls.append(neg_log_likelihood)

    #     prev_end_loc = end_loc
    #     if end_loc == seq_len:
    #         break

    # ppl = torch.exp(torch.stack(nlls).mean())
    # return ppl
