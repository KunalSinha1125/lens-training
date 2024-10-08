import os
from pathlib import Path
from typing import Any, List, Optional

import huggingface_hub
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, IterableDataset, load_dataset
from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    CLIPModel,
    CLIPProcessor
)
from accelerate import Accelerator
from utils import (
    create_dataloader,
    create_prompt_sample,
    create_sampler,
    default_device,
    blip_prompt
)
import numpy as np
import random
import json
random.seed(1)

def flatten(l):
    return [item for sublist in l for item in sublist]

CACHE_DIR = "/nlp/scr/ksinha2/JUICE-SCR/my_model_dir"

class Lens(nn.Module):
    def __init__(
        self,
        clip_name: str = None,#"hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        blip_name: str = "Salesforce/blip-image-captioning-large",#"unography/blip-large-long-cap",#"Salesforce/blip-image-captioning-large",
        attributes_weights: str = "zw_attributes_laion_ViT_H_14_2B_descriptors_text_davinci_003_full.pt",
        tags_weights: str = "zw_tags_laion_ViT_H_14_2B_vocab_lens.pt",
        vocab_attributes: str = "llm-lens/descriptors-text-davinci-003",
        vocab_tags: str = "llm-lens/vocab_tags",
        split_attributes: str = "full",
        split_tags: str = "train",
        #num_total_tags: int = 5000,
        load_8bit: bool = False,
        device: torch.device = default_device,
    ):
        super().__init__()
        # Load Base model
        accelerator = Accelerator()
        self.device = device
        self.clip_name = clip_name
        self.blip_name = blip_name
        if self.clip_name is not None:
            print("Before CLIP model")
            self.clip_model = self.load_clip_model(self.clip_name, self.device)
            print("After CLIP model")
            # Load weights
            #huggingface_hub.hf_hub_download(
            #    repo_id="llm-lens/attributes",
            #    filename=attributes_weights,
            #    local_dir=str(Path(Path(__file__).resolve().parent) / "weights"),
            #)
            print("Before tags download")
            huggingface_hub.hf_hub_download(
                repo_id="llm-lens/tags",
                filename=tags_weights,
                local_dir=str(Path(Path(__file__).resolve().parent) / "weights"),
            )
            print("After tags download")

            #self.attributes_weights = torch.load(
            #    str(
            #        Path(Path(__file__).resolve().parent)
            #        / f"weights/{attributes_weights}"
            #    ),
            #    map_location=self.device,
            #).float()
            self.tags_weights = torch.load(
                str(Path(Path(__file__).resolve().parent) / f"weights/{tags_weights}"),
                map_location=self.device,
            ).float()
            # Load Vocabularies
            print("Before vocab tags")
            self.vocab_tags = np.array(load_dataset(vocab_tags, split=split_tags, cache_dir=CACHE_DIR)["prompt_descriptions"])
            print("After vocab tags")
            #tags_indices = random.sample(list(range(len(self.vocab_tags))), num_total_tags)
            #self.tags_weights = self.tags_weights[:, torch.tensor(tags_indices).to(device)]
            #self.vocab_tags = self.vocab_tags[tags_indices]
            #print("Before tags tokens")
            #self.tags_tokens = open_clip.tokenize(self.vocab_tags).to(device)
            #print("After tags tokens")
            #token_len = self.tags_tokens.argmin(dim=-1).max().item()
            #self.tags_tokens = self.tags_tokens[:, :token_len]
            #self.clip_model.context_length = token_len
            #self.vocab_attributes = flatten(
            #    load_dataset(vocab_attributes, split=split_attributes)[
            #        "prompt_descriptions"
            #    ]
            #)

        if self.blip_name is not None:
            print("Before blip2: ", torch.cuda.mem_get_info()[0] / 1e9)
            self.blip_model = self.load_caption_model(
                self.blip_name, load_8bit, self.device
            )
            print("Before blip2 processor: ", torch.cuda.mem_get_info()[0] / 1e9)
            self.blip_processor = AutoProcessor.from_pretrained(self.blip_name, cache_dir=CACHE_DIR)

    def load_caption_model(
        self, model_name: str, load_8bit: bool, device: torch.device
    ):
        if load_8bit:
            model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if device == "cpu" else torch.float16,
                device_map={"": device},
                load_in_8bit=True,
                config="blip_config.json",
                cache_dir=CACHE_DIR
            )
        else:
            model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if device == "cpu" else torch.float16,
                config="blip_config.json",
                cache_dir=CACHE_DIR
            )
        model = model.train()
        model = model.to(device)

        return model

    def load_clip_model(self, model_name: str, device: torch.device):
        if "openai" in model_name:
            model = CLIPModel.from_pretrained(model_name).to(device)
        elif "laion" in model_name:
            print("Before create model and transform")
            print(f"Memory left: {torch.cuda.mem_get_info()[0] / 1e9}")
            model = open_clip.create_model_and_transforms(model_name)[0].to(device)
            print(f"Memory left: {torch.cuda.mem_get_info()[0] / 1e9}")
            print("After create model and transform")
        model = model.train()
        model.set_grad_checkpointing()
        return model

    def __call__(
        self,
        clip_image=None,
        blip_image=None,
        blip_input_ids=None,
        num_tags: int = 10,
        num_attributes: int = 1,
        contrastive_th: float = 0.2,
        max_length: int = 5,
        min_length: int = 1,
        top_k: int = 1,
        questions = [],
        num_captions: int = 5,
        return_tags: bool = False,
        return_attributes: bool = False,
        return_global_caption: bool = False,
        return_intensive_captions: bool = False,
        return_prompt: bool = False,
    ):

        samples = {
            "clip_image": clip_image,
            "blip_image": blip_image,
            "blip_input_ids": blip_input_ids
        }
        if return_tags:
            samples = self.forward_tags(
                samples, num_tags=num_tags, contrastive_th=contrastive_th
            )
        if return_attributes:
            samples = self.forward_attributes(
                samples, num_attributes=num_attributes, contrastive_th=contrastive_th
            )
        if return_global_caption:
            samples = self.forward_caption(
                samples,
                num_captions=num_captions,
                max_length=max_length,
                min_length=min_length,
            )
        if return_intensive_captions:
            samples = self.forward_intensive_caption(
                samples,
                max_length=max_length,
                min_length=min_length,
                top_k=top_k,
                num_captions=num_captions,
            )

        if questions:
            samples["questions"] = questions
        if return_prompt:
            mode = "vqa"
            #if return_tags and not return_attributes:
                #mode = "tags_only"
            #elif return_attributes and not return_tags:
            #    mode = "attributes_only"
            samples = self.create_prompt_from_samples(samples, mode=mode)

        return samples

    def forward_tags(
        self, samples: dict, num_tags: int = 100, contrastive_th: float = 0.2
    ):
        # Get Image Features
        tags = []
        try:
            image_features = self.clip_model.encode_image(samples["clip_image"].to("cuda"))#.to(dtype=torch.bfloat16))
        except:
            image_features = self.clip_model.get_image_features(
                pixel_values=samples["clip_image"]
            ) 
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        #with torch.no_grad():
        #    text_features = self.clip_model.encode_text(self.tags_tokens)
        text_features_norm = self.tags_weights / self.tags_weights.norm(dim=-1, keepdim=True)
        text_scores = (image_features_norm @ text_features_norm).float()
        print("After computing text scores")
        #text_scores = (image_features_norm @ self.tags_weights).float()
        top_scores, top_indexes = text_scores.topk(k=num_tags, dim=-1)
        #bsz, k = top_indexes.shape
        #chosen_text_features = self.clip_model.encode_text(self.tags_tokens[top_indexes].reshape((bsz*k, -1)))
        #chosen_text_features_norm = chosen_text_features / chosen_text_features.norm(dim=-1, keepdim=True)
        #top_scores = torch.matmul(chosen_text_features_norm.reshape((bsz, k, -1)), image_features.unsqueeze(-1)).float().squeeze()
        for scores, indexes in zip(top_scores, top_indexes):
            #filter_indexes = indexes[scores >= contrastive_th]
            #if len(filter_indexes) > 0:
            top_k_tags = [self.vocab_tags[index] for index in indexes]
            #else:
            #    top_k_tags = []
            tags.append(top_k_tags)
        samples[f"tags"] = tags
        samples[f"top_scores_tags"] = top_scores
        return samples

    def forward_attributes(
        self, samples: dict, num_attributes: int = 250, contrastive_th: float = 0.2
    ):
        # Get Image Features
        attributes = []
        try:
            image_features = self.clip_model.encode_image(
                samples["clip_image"].to(self.device)
            )
        except:
            image_features = self.clip_model.get_image_features(
                pixel_values=samples["clip_image"]
            )
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_scores = image_features_norm @ self.attributes_weights
        top_scores, top_indexes = text_scores.float().cpu().topk(
            k=num_attributes if num_attributes else len(text_scores.squeeze()), dim=-1
        )
        for scores, indexes in zip(top_scores, top_indexes):
            #filter_indexes = indexes[scores >= contrastive_th]
            #if len(filter_indexes) > 0:
            #    top_k_tags = [self.vocab_attributes[index] for index in filter_indexes]
            #else:
            #    top_k_tags = []
            top_k_tags = [self.vocab_attributes[index] for index in indexes]
            attributes.append(top_k_tags)
        samples[f"attributes"] = attributes
        samples[f"top_scores_attributes"] = top_scores
        return samples

    def forward_caption(
        self,
        samples: dict,
        num_captions: int = 10,
        max_length: int = 30,
        min_length: int = 10,
    ):
        # Beam search
        captions_list = []
        pixel_values = samples["blip_image"].to(self.device, self.blip_model.dtype)
        input_ids = samples["blip_input_ids"].to(self.device)
        captions_outputs = self.blip_model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            do_sample=False,
            num_return_sequences=num_captions,
            num_beams=num_captions,
            top_p=1,
            max_length=max_length,
            min_length=min_length,
            output_scores=True,
            return_dict_in_generate=True
        )
        captions_ids = captions_outputs.sequences
        samples["top_scores_captions"] = captions_outputs.sequences_scores

        captions = self.blip_processor.batch_decode(
            captions_ids, skip_special_tokens=True
        )
        for caption in captions:
            captions_list.append(caption[12:].strip())
        captions_list = np.array(captions_list).reshape((-1, num_captions))
        samples["captions"] = captions_list
        return samples

    def forward_intensive_caption(
        self,
        samples: dict,
        max_length: int = 50,
        min_length: int = 1,
        top_k: int = 5,
        num_captions: int = 10,
    ):
        pixel_values = samples["blip_image"].to(self.device, self.blip_model.dtype)
        input_ids = samples["blip_input_ids"].to(self.device)
        bsz, _ = input_ids.shape
        captions_output = self.blip_model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_captions,
            num_return_sequences=num_captions,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
            temperature=1.0,
        )
        sequences = captions_output.sequences
        #captions_logits = self.blip_model.compute_transition_scores(sequences, scores)

        captions_text = self.blip_processor.batch_decode(captions_output.sequences, skip_special_tokens=True)
        captions_text = np.array([cap[len(blip_prompt):].strip() for cap in captions_text]).reshape(bsz, -1)
        samples["intensive_captions"] = captions_text
        attention_mask = torch.logical_and(
            sequences != self.blip_processor.tokenizer.pad_token_id,
            sequences != self.blip_processor.tokenizer.sep_token_id
        ).int()
        logits = self.blip_model(
            pixel_values=pixel_values.repeat_interleave(num_captions, dim=0), 
            input_ids=sequences,
            attention_mask=attention_mask
        ).logits
        logprobs = logits.log_softmax(dim=-1).max(dim=-1).values
        #logprobs[attention_mask == 0] = 0
        #z = (attention_mask != 0 ).sum(dim=-1)
        captions_scores = (logprobs.sum(dim=-1)).reshape((bsz, num_captions)) 
        #labels = sequences.masked_fill(attention_mask == 0, -100).to(self.device)
        #loss = F.cross_entropy(
        #    logits.reshape(-1, logits.shape[-1]),
        #    labels.view(-1),
        #    ignore_index=-100,
        #    reduction='none'
        #).reshape((bsz, num_captions, -1))
        #z = (labels.reshape((bsz, num_captions, -1)) > -1).sum(dim=-1)
        #blip_perplexity = -loss.sum(dim=-1) / z
        samples["top_scores_intensive_captions"] = captions_scores
        return samples

    # This function could be more efficient
    def create_prompt_from_samples(
        self,
        samples: dict,
        mode: str = "all",  # vqa or vision or hm or or all
    ):
        num_samples = samples["blip_image"].shape[0]
        if "intensive_captions" in samples:
            num_samples = len(samples["intensive_captions"])
        prompts = []
        for idx in range(num_samples):
            prompt = create_prompt_sample(samples, idx, desc_idx=0, mode=mode)

            prompts.append(prompt)
        samples["prompts"] = prompts
        return samples

    def hf_dataset_transform(
        self,
        ds: Dataset,
        processor: "LensProcessor",
        num_tags: int = 5,
        num_attributes: int = 5,
        contrastive_th: float = 0.2,
        num_beams: int = 5,  # For beam search
        max_length: int = 30,
        min_length: int = 10,
        top_k: int = 50,
        num_captions: int = 10,
        return_tags: bool = True,
        return_attributes: bool = True,
        return_global_caption: bool = True,
        return_intensive_captions: bool = True,
        distributed_sampling: bool = False,
        batch_size: int = 8,
        num_workers: int = 0,
    ):
        dataset = LensDataset(ds, None, processor)
        # Create sampler
        sampler = create_sampler(dataset, distributed=distributed_sampling)
        # Create Dataloader
        dataloader = create_dataloader(
            dataset, sampler, batch_size=batch_size, num_workers=num_workers
        )

        # Get tags, attributes, caption, intensive_captions
        result = []
        for batch in dataloader:
            with torch.no_grad():
                batch = self(
                    batch,
                    num_tags=num_tags,
                    num_attributes=num_attributes,
                    contrastive_th=contrastive_th,
                    num_beams=num_beams,  # For beam search
                    max_length=max_length,
                    min_length=min_length,
                    top_k=top_k,
                    num_captions=num_captions,
                    return_tags=return_tags,
                    return_attributes=return_attributes,
                    return_global_caption=return_global_caption,
                    return_intensive_captions=return_intensive_captions,
                )

                keys = [
                    key
                    for key in batch.keys()
                    if key
                    in ["id", "tags", "attributes"]#, "caption", "intensive_captions"]
                ]
                # print(f"keys: {keys}")
                for tuples in zip(*[batch[key] for key in keys]):
                    result.append(
                        {
                            k: (v.item() if k == "id" else v)
                            for k, v in zip(keys, tuples)
                        }
                    )

        if distributed_sampling is False:
            # To-Do: Add new columns to the huggingface dataset
            dict_ = {}
            for res in result:
                dict_[res["id"]] = {k: v for k, v in res.items() if k != "id"}

            # Map new columns would be faster
            def add_info(example):
                for component in [
                    "tags",
                    "attributes",
                    # "caption",
                    # "intensive_captions",
                ]:
                    try:
                        example[component] = dict_[example["id"]][component]
                    except:
                        pass
                return example

            result_ds = ds.map(add_info, batched=False)
            return result_ds
        else:
            # Only return the new componenets
            result_ds = Dataset.from_dict(
                {key: [d[key] for d in result] for key in result[0]}
            )
            return result_ds


class LensProcessor:
    def __init__(
        self,
        clip_name: str = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        blip_name: str = "Salesforce/blip-image-captioning-large",
    ):
        print("Before CLIP processor")
        self.clip_processor = self.load_clip_transform(clip_name)
        print("After CLIP processor")
        self.blip_processor = AutoProcessor.from_pretrained(blip_name, cache_dir=CACHE_DIR)

    def load_clip_transform(self, model_name: str):
        if "openai" in model_name:
            return CLIPProcessor.from_pretrained(model_name, cache_dir=CACHE_DIR)

        elif "laion" in model_name:
            return open_clip.create_model_and_transforms(model_name)[2]

    def __call__(self, images):
        try:
            clip_image = torch.stack([self.clip_processor(image) for image in images])
        except:
            clip_image = self.clip_processor(images=images, return_tensors="pt")["pixel_values"]
        outputs = self.blip_processor(
            images=images, text=[blip_prompt] * len(images), return_tensors="pt"
        )
        blip_image = outputs["pixel_values"]
        blip_input_ids = outputs["input_ids"]
        return {
            "clip_image": clip_image,
            "blip_image": blip_image,
            "blip_input_ids": blip_input_ids
        }


class LensDataset(IterableDataset):
    def __init__(
        self,
        ds: Dataset,
        processor: Optional[LensProcessor] = None,
        ds_name: str = "cifar10",
        task: str = "vqa"
    ):
        self.ds = ds
        self.processor = processor
        self.ds_name = ds_name
        self.task = task
        label_dir = "labels"
        if task == "classification":
            classes_dir = os.path.join(label_dir, f"{ds_name}.json")
            with open(classes_dir, 'r') as f:
                self.classes = json.load(f)
                self.classes = [cl.replace("_", " ") for cl in self.classes]

    def __iter__(self):
        img_key, label_key = "image", "label"
        question_key = "question"
        question_type_key = "question_type"
        if self.ds_name == "cifar10":
            img_key = "img"
        elif "VQA" in self.ds_name:
            label_key = "multiple_choice_answer"
        for elem in self.ds:
            processed = self.processor([elem[img_key]])
            clip_image = processed["clip_image"]
            blip_image, blip_input_ids = processed["blip_image"], processed["blip_input_ids"]
            label = None
            if self.task == "vqa":
                label = elem[label_key]
            elif self.task == "classification":
                label = self.classes[int(elem[label_key])]
            question = elem[question_key] if question_key in elem else ""
            question_type = elem[question_type_key] if question_type_key in elem else ""
            yield clip_image.squeeze(), blip_image.squeeze(), blip_input_ids.squeeze(), question, question_type, label
            
    def __len__(self):
        return len(self.ds) 

    # def __getitem__(self, idx):
    #     img_key, label_key = "image", "label"
    #     if self.ds_name == "cifar10":
    #         img_key = "img"
    #     image = self.ds[idx][img_key]
    #     label = self.classes[int(self.ds[idx][label_key])]
    #     clip_image = self.processor([image])
    #     return clip_image.squeeze(), label
