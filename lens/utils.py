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

blip_prompt = "a picture of"

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
    caption_col="captions",
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
        context = ".".join(
            samples[intensive_captions_col][idx]
        )
        prompt = f"Context: {context.lower()}\n\nQuestion: {question.lower()}\n\nAnswer:"
        #prompt += "Captions:"
        #prompt += ".".join(
        #    samples[intensive_captions_col][idx][:num_intensive_captions]
        #)
        #prompt += "\nQuestion:"
        #prompt += question
        #prompt += "\nAnswer with a word or short phrase. Answer:"

    elif mode == "vqa_single":
        #prompt += f"Image caption: {samples[intensive_captions_col][idx][desc_idx]}.\nQuestion: {question.lower()}\nShort Answer:"
        context = samples[intensive_captions_col][idx][desc_idx]
        #prompt += f"Answer the question based on information in the context. Context: {context}. Question: {question} Answer:"
        #prompt += f"Context: {context.lower()}\n\nQuestion: {question}\n\nAnswer:"
        prompt = f"Read this and answer the question\n\n{context}.\n\n{question}"
        #prompt = f"Article: {context}.\n\nNow answer this question: {question}"
        print(prompt)
        #prompt += f"{question} Output a word or phrase to answer this question based on the caption."
        #prompt += "\nOutput:"

    elif mode == "baseline":
        prompt = f"Question: {question}\nShort Answer:"

    elif mode == "baseline_gemma":
        prompt = f"Question: {question}\nOutput no more than a few words to answer the question succintly. Answer:"

    elif mode == "attributes_and_captions":
        attributes = ". ".join(samples[attributes_col][idx])
        prompt += f"Attributes: {attributes}"
        captions = ". ".join(samples[intensive_captions_col][idx])
        prompt += f"\nCaptions: {captions}"
        prompt += f"\nQuestion: {question}"
        prompt += "\nShort Answer: "

    elif mode == "vision":
        prompt += "Tag: "
        prompt += ",".join(samples[tags_col][idx])
        prompt += "\nAttributes: "
        prompt += ",".join(samples[attributes_col][idx])
        prompt += "\nQuestion:"
        prompt += question
        prompt += "\nShort Answer:"

    elif mode == "tags_only":
        tags = ",".join(samples[tags_col][idx])
        prompt += f"Tags: {tags}"
        prompt += f"\nQuestion: {question}"
        prompt += f"\nShort Answer:"
        #prompt += f"\"Instruct: given the image tags {tags}, output a word naming the object in the image. Output: \""

    elif mode == "tags_only_single":
        tag = samples[tags_col][idx][desc_idx].lower()
        prompt += f"\"Instruct: given the image tag {tag}, output a word naming the object in the image. Output: \""

    elif mode == "tags_only_test":
        tags = samples[tags_col][idx][desc_idx]
        #loop_tags = ",".join(tags[:desc_idx] + tags[desc_idx+1:])
        prompt += f"Instruct: you are given the image tags {tags}. Based on this information, output a word that describes the image. \nOutput:"
        #prompt += "\nTag: canine. Answer: dog"
        #prompt += "\nTag: cedar. Answer: tree"
        #prompt += "\nTag: driving. Answer: car"
        #prompt += f"\nTag: {tags[desc_idx]}. Answer: "
        #prompt += f"\nTags: {loop_tags}. Answer: "
    
    elif mode == "tags_only_test_loop":
        tags = samples[tags_col][idx]
        loop_tags = ",".join(tags[:desc_idx] + tags[desc_idx+1:])
        prompt = f"Instruct: you are given the image tags {loop_tags}. Based on this information, output a word that describes the image. \nOutput:"

    elif mode == "tags_only_loop":
        tags = samples[tags_col][idx]
        loop_tags = ",".join(tags[:desc_idx] + tags[desc_idx+1:])
        prompt = (
            f'''Provide one word to describe the image based on the information in the image tags. 
            Tags: claws,crustacean,red,shell,underwater
            Short Answer: crab
            Tags: {loop_tags}
            Short Answer: '''
        )
        # tags = samples[tags_col][idx]
        # prompt += "\nTag: "
        # prompt += ",".join(tags[:desc_idx] + tags[desc_idx+1:])
        # prompt += "\nQuestion: "
        # prompt += question
        # prompt += "\nShort Answer: "

    elif mode == "tags_only_single_phi2":
        tag = samples[tags_col][idx][desc_idx]
        #prompt = f"Instruct: Describe the image whose tag is {tag}.\nOutput:"
        #prompt = f'''Alice: can you tell me what you see in this image? The image tags are [{tag}].\nBob:'''
        prompt = f"Describe the given image."

    elif mode == "tags_only_vqa":
        tags = samples[tags_col][idx]
        tags_formatted = ", ".join(tags)
        prompt = f"\"Instruct: {question} Given the image tags {tags_formatted}, output one word to answer this question. Output: \""

    elif mode == "tags_only_vqa_loop":
        tags = samples[tags_col][idx]
        loop_tags = ",".join(tags[:desc_idx] + tags[desc_idx+1:])
        prompt = f"\"Instruct: {question} Given the image tags {loop_tags}, output one word to answer this question. Output: \""
    
    elif mode == "captions_only_vqa":
        captions = samples[caption_col][idx]
        captions = "\n".join([cap.strip() for cap in captions])
        prompt = f"\"Instruct: Answer the question based on the image captions.\nCaptions:\n{captions}\nQuestion: {question} Answer in one word, phrase, or number.\nOutput: \""

    elif mode == "captions_only_vqa_single":
        caption = samples[caption_col][idx][desc_idx]
        prompt = f"\"Instruct: Answer the question based on the image caption.\nCaption: {caption}\nQuestion: {question} Answer in one word, phrase, or number.\nOutput: \""

    elif mode == "intensive_captions_only_vqa":
        captions = samples[intensive_captions_col][idx]
        captions = ".".join(captions) 
        #prompt = f"\"Instruct: Answer the question based on the image captions.\nCaptions:\n{captions}\nQuestion: {question} Answer in one word, phrase, or number.\nOutput: \""  
    
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


