from model import Lens, LensDataset, LensProcessor, CACHE_DIR
import requests
from PIL import Image
from scipy.special import rel_entr
from transformers import Trainer, TrainingArguments, CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch
import numpy as np
from utils import create_prompt_sample, create_dataloader, create_sampler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
import wandb
import matplotlib.pyplot
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
import torch.autograd.profiler as profiler
import torch.nn as nn
from accelerate import Accelerator
from evaluate import compute_class_acc, compute_vqa_acc
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"You are using device {device}")
print(torch.cuda.mem_get_info()[0] / 1e9)
accelerator = Accelerator()
print(torch.cuda.mem_get_info()[0] / 1e9)
lens = Lens()
processor = LensProcessor()
print(torch.cuda.mem_get_info()[0] / 1e9)
llm_name = "google/flan-t5-xl"
llm_model = T5ForConditionalGeneration.from_pretrained(
    llm_name, trust_remote_code=True, 
    cache_dir=CACHE_DIR).to(device)
tokenizer = T5Tokenizer.from_pretrained(llm_name, trust_remote_code=True, cache_dir=CACHE_DIR)
IGNORE_INDEX = -100

def compute_llm_likelihood(samples, labels, gamma=1e-1, desc="tags"):
    bsz, k = np.array(samples[desc]).shape
    num_inputs = bsz * k
    #inputs, all_labels = [], [] 
    prompts = []
    for i in range(bsz):
        for j in range(k):
            prompt = create_prompt_sample(
                samples, i, desc_idx=j, mode=f"vqa_single",
            )
            prompts.append(prompt)
    # Tokenize full inputs
    tokenizer.padding_side = "left"
    prompt_tokens = tokenizer(prompts, return_tensors="pt", add_special_tokens=True, padding=True).to(device)
    reader_tok, reader_mask = prompt_tokens.input_ids[:, :-1], prompt_tokens.attention_mask[:, :-1] 
    tokenizer.padding_side = "right"
    label_tokens = tokenizer(labels, return_tensors="pt", add_special_tokens=True, padding=True).to(device)
    answer_tok, answer_mask = label_tokens.input_ids[:, :-1], label_tokens.attention_mask[:, :-1]
    repeat_answer_tok = torch.repeat_interleave(answer_tok[:, None], k, dim=-1).view(-1, answer_tok.shape[-1])
    repeat_answer_mask = torch.repeat_interleave(answer_mask[:, None], k, dim=-1).view(-1, answer_mask.shape[-1])

    lsr_logits = llm_model(
        input_ids=reader_tok,
        attention_mask=reader_mask,
        decoder_input_ids=repeat_answer_tok,
        decoder_attention_mask=repeat_answer_mask,
        use_cache=False,
    ).logits

    lsr_labels = repeat_answer_tok.masked_fill(repeat_answer_mask == 0, IGNORE_INDEX).to(device)
    token_loss = F.cross_entropy(
        lsr_logits.reshape(-1, lsr_logits.shape[-1]),
        lsr_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction='none',
    ).reshape((bsz, k, -1))
    z = (lsr_labels.reshape((bsz, k, -1)) > -1).sum(dim=-1)
    llm_perplexity = -token_loss.sum(dim=-1) / z
    llm_likelihood = torch.softmax(llm_perplexity / gamma, dim=-1)
    return llm_likelihood, llm_perplexity

def compute_llm_likelihood_hf(samples, labels, gamma=1e-2, desc="tags"):
    bsz, k = np.array(samples[desc]).shape
    prompts = []
    for i in range(bsz):
        for j in range(k):
            prompt = create_prompt_sample(
                samples, i, desc_idx=j, mode=f"{desc}_only_test",
                question_prompt=samples["questions"][i]
            )
            prompts.append(f"{prompt} {labels[i]}")
    print(torch.cuda.mem_get_info()[0] / 1e9)
    results = perplexity.compute(predictions=prompts, model_id='microsoft/phi-2', add_start_token=False, device="cuda")
    lm_perplexity = torch.tensor(results['perplexities']).reshape((bsz, k)).to(device) # see above
    lm_likelihood = torch.softmax(lm_perplexity / gamma, dim=-1)
    print(torch.cuda.mem_get_info()[0] / 1e9)
    return lm_likelihood, lm_perplexity

# def compute_captions_likelihood(samples, gamma=1.0):
#     bsz, k = np.array(samples["intensive_captions"]).shape
#     logits = samples["intensive_captions_logits"]
#     logprobs = logits.log_softmax(dim=-1)
#     nll = -logprobs.mean(dim=-1)
#     perplexity = torch.exp(-nll).reshape(bsz, k)
#     likelihood = torch.softmax(perplexity / gamma, dim=-1)
#     return likelihood, perplexity

def compute_tags_likelihood(samples, gamma=1e-1, desc="tags"):
    bsz, k = np.array(samples[desc]).shape
    tags_scores = samples[f"top_scores_{desc}"].reshape((bsz, k)).to(device)
    tags_likelihood = torch.softmax(tags_scores / gamma, dim=-1)
    return tags_likelihood, tags_scores

def compute_loss(samples, labels, table_name=None, desc="tags"):
    tags_likelihood, tags_scores = compute_tags_likelihood(samples, desc=desc)
    with torch.no_grad():
        llm_likelihood, llm_perplexity = compute_llm_likelihood(samples, labels, desc=desc)
    if table_name:
        table_data = {
            "Description likelihood": tags_likelihood[0],
            "Description scores": tags_scores[0],
            "LM likelihood": llm_likelihood[0],
            "LM perplexity": llm_perplexity[0],
        }
        indices = llm_perplexity[0].argsort(descending=True).detach().cpu().numpy()
        for k, v in table_data.items():
            table_data[k] = v.detach().to(dtype=torch.float16).cpu().numpy()[indices]
        table_data[desc] = np.array(samples[desc][0])[indices]
        table = wandb.Table(data=pd.DataFrame(table_data))
        wandb.log({table_name: table})
    kl_penalty = F.kl_div(
        tags_likelihood.log(), llm_likelihood.log(),
        reduction="batchmean", log_target=True
    )
    print("BLIP scores: ", tags_scores[0].sort().values)
    print("BLIP likelihood: ", tags_likelihood[0].sort().values)
    print("LLM perplexity: ", llm_perplexity[0].sort().values)
    print("LLM likelihood: ", llm_likelihood[0].sort().values)
    print("Loss: ", kl_penalty.item())
    import pdb; pdb.set_trace()
    return kl_penalty

def forward(clip_image, blip_image, blip_input_ids, questions):
    print("Entering forward pass")
    samples = lens(
        clip_image,
        blip_image,
        blip_input_ids,
        return_tags=False,
        return_attributes=False,
        return_intensive_captions=True,
        return_global_caption=False,
        return_prompt=True,
        questions=questions
    )
    print("Memory left: ", torch.cuda.mem_get_info()[0] / 1e9)
    print("Completed forward pass")
    return samples

def main(train_name, train_split, val_name, val_split, task, desc,
         num_epochs=100, lr=1e-3, batch_size=4, train_size=8000, val_size=800):
    wandb.init(project="lens-training-coco-dataset")
    save_path = "trained_model_attributes.pt"
    train_ds_raw = load_dataset(train_name, split=train_split, streaming=True, trust_remote_code=True, cache_dir=CACHE_DIR)
    train_ds_raw = train_ds_raw.shuffle(seed=0, buffer_size=10000)
    train_ds = LensDataset(train_ds_raw, processor, train_name)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size)
    print("Created train loader")
    val_ds_raw = load_dataset(val_name, split=val_split, streaming=True, trust_remote_code=True, cache_dir=CACHE_DIR)
    val_ds_raw = val_ds_raw.shuffle(seed=0, buffer_size=10000)
    val_ds = LensDataset(val_ds_raw, processor, val_name)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size)
    print("Created val loader")
    params = lens.blip_model.parameters() if desc == "intensive_captions" else lens.clip_model.parameters()
    optimizer = torch.optim.Adam(params, lr=lr)
    #print("Before prepare")
    #lens.clip_model = accelerator.prepare(lens.clip_model)
    #print("After prepare")

    for epoch in range(num_epochs):
        #Compute train loss
        train_loss_epoch = 0
        correct = 0
        for i, (clip_image, blip_image, blip_input_ids, questions, question_types, labels) in enumerate(train_dataloader):
            if i >= (train_size // batch_size):
                continue
            samples = forward(clip_image, blip_image, blip_input_ids, questions)
            train_loss = compute_loss(samples, labels, desc=desc)
            wandb.log({"train_loss": train_loss.item()})
            train_loss_epoch += train_loss.item()
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                if task == "classification":
                    correct += compute_class_acc(samples["prompts"], labels, llm_model, tokenizer, train_ds.classes)
                elif task == "vqa":
                    correct += compute_vqa_acc(samples["prompts"], labels, llm_model, tokenizer, llm_name)
            print(f"Finished batch {i}")
        wandb.log({"train_loss_epoch": train_loss_epoch / (train_size // batch_size)})
        wandb.log({"train_acc": correct / train_size})
        #Compute val loss
        val_loss_epoch = 0
        correct = 0
        for i, (clip_image, blip_image, blip_input_ids, questions, question_types, labels) in enumerate(val_dataloader):
            if i >= (val_size // batch_size):
                continue
            with torch.no_grad():
                samples = forward(clip_image, blip_image, blip_input_ids, questions)
                val_loss = compute_loss(samples, labels, desc=desc).item()
                wandb.log({"val_loss": val_loss})
                val_loss_epoch += val_loss
                if task == "classification":
                    correct += compute_class_acc(samples["prompts"], labels, llm_model, tokenizer, train_ds.classes)
                elif task == "vqa":
                    correct += compute_vqa_acc(samples["prompts"], labels, llm_model, tokenizer)
        wandb.log({"val_loss_epoch": val_loss_epoch / (val_size // batch_size)})
        wandb.log({"val_acc": correct / val_size})

if __name__ == "__main__":
    '''
    imagenet-1k: python3 train_alt.py --train_dataset imagenet-1k --train_split validation --val_split test --desc tags
    food101: python3 train_alt.py --train_dataset food101 --train_split train --val_split validation --desc tags
    vqav2: python3 train_alt.py --train_dataset HuggingFaceM4/VQAv2 --train_split train --val_split validation --task vqa --desc intensive_captions
    vqav2: python3 train_alt.py --train_dataset ReplugLens/VQAv2 --train_split minival_validation --val_split testdev --task vqa --desc intensive_captions
    '''
    parser = ArgumentParser(description='Train',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_dataset',
                        default="cifar10",
                        choices=["cifar10", "imagenet-1k", "food101", "HuggingFaceM4/VQAv2", "ReplugLens/VQAv2"],
                        type=str,
                        help='Name of train dataset?')
    parser.add_argument('--train_split',
                        default="train",
                        type=str,
                        help='Name of split for train dataset?')
    parser.add_argument('--val_dataset',
                        default=None,
                        choices=["cifar10", "imagenet-1k", "food101", "HuggingFaceM4/VQAv2"],
                        type=str,
                        help='Name of val dataset?')
    parser.add_argument('--val_split',
                        default="test",
                        type=str,
                        help='Name of split for val dataset?')
    parser.add_argument('--task',
                        default="classification",
                        type=str,
                        choices=["classification", "vqa"],
                        help='Type of dataset?')
    parser.add_argument('--desc',
                        type=str,
                        default="captions",
                        choices=["tags", "attributes", "captions", "intensive_captions"],
                        help='Which type of description to include in prompt and to train')
    args = parser.parse_args()
    train_name, train_split, val_split = args.train_dataset, args.train_split, args.val_split
    val_name = args.val_dataset
    if not val_name:
        val_name = train_name
    main(train_name, train_split, val_name, val_split, task=args.task, desc=args.desc)
