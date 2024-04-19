from model import Lens, LensDataset, LensProcessor
import requests
from PIL import Image
from scipy.special import rel_entr
from transformers import Trainer, TrainingArguments, CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoTokenizer
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
from evaluate import compute_class_acc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"You are using device {device}")
print(torch.cuda.mem_get_info()[0] / 1e9)
accelerator = Accelerator()
print(torch.cuda.mem_get_info()[0] / 1e9)
lens = Lens()
processor = LensProcessor()
print(torch.cuda.mem_get_info()[0] / 1e9)
llm_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
IGNORE_INDEX = -100

def compute_llm_likelihood(samples, labels, gamma=1.0, desc="tags"):
    bsz, k = np.array(samples[desc]).shape
    num_inputs = bsz * k
    #inputs, all_labels = [], [] 
    prompts = []
    for i in range(bsz):
        for j in range(k):
            prompt = create_prompt_sample(
                samples, i, desc_idx=j, mode=f"{desc}_only_single",
            )
            prompts.append(prompt)
            #inputs.append(f"{prompt} {labels[i]}")
            #all_labels.append(labels[i])
    # Tokenize full inputs
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    #lsr_tokens = tokenizer(inputs, return_tensors="pt", padding=True).to(device)
    #lsr_input_ids, lsr_attention_mask = lsr_tokens.input_ids, lsr_tokens.attention_mask
    #lsr_input_ids = torch.cat([lsr_input_ids, torch.full((num_inputs, 1), tokenizer.eos_token_id).to(device)], dim=-1)
    #lsr_attention_mask = torch.cat([lsr_attention_mask, torch.zeros((num_inputs, 1)).to(device)], dim=-1)
    # Tokenize answers
    #label_tokens = tokenizer(all_labels, return_tensors="pt").to(device)
    #label_input_ids, label_attention_mask = label_tokens.input_ids, label_tokens.attention_mask
    #input_len, label_len = lsr_input_ids.shape[-1], label_input_ids.shape[-1]

    tokenizer.padding_side = "left"
    prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True)
    tokenizer.padding_side = "right"
    label_tokens = tokenizer(labels, return_tensors="pt", padding=True, add_special_tokens=True)
    reader_tok, reader_mask = prompt_tokens.input_ids, prompt_tokens.attention_mask
    answer_tok, answer_mask = label_tokens.input_ids, label_tokens.attention_mask

    repeat_answer_tok = torch.repeat_interleave(answer_tok[:, None], k, dim=1).view(-1, answer_tok.shape[-1])
    repeat_answer_mask = torch.repeat_interleave(answer_mask[:, None], k, dim=1).view(-1, answer_mask.shape[-1])
    reader_tok = reader_tok.reshape(-1, reader_tok.shape[-1])
    reader_mask = reader_mask.reshape(-1, reader_mask.shape[-1])

    lsr_input_ids = torch.cat((reader_tok, repeat_answer_tok), dim=-1).to(device)
    lsr_attention_mask = torch.cat((reader_mask, repeat_answer_mask), dim=-1).to(device)
    with torch.autocast("cuda"):
        with torch.no_grad():
            lsr_logits = llm_model(
                input_ids=lsr_input_ids[:, :-1],
                attention_mask=lsr_attention_mask[:, :-1],
                use_cache=False,
            ).logits

    # compute perplexity of question
    continuation_length = repeat_answer_tok.shape[-1]
    lsr_logits = lsr_logits[:, -continuation_length:]
    lsr_labels = repeat_answer_tok.masked_fill(repeat_answer_mask == 0, IGNORE_INDEX).to(device)
    #vocab_size = lsr_logits.shape[-1]
    #lsr_end_indices = (lsr_attention_mask).argmin(axis=1).squeeze()
    #lsr_end_indices[lsr_end_indices == 0] = input_len
    #lsr_labels = label_attention_mask
    #TODO: optimize
    #label_logits = torch.zeros((num_inputs, label_len, vocab_size)).to(device)
    #for i in range(num_inputs):
    #    end = lsr_end_indices[i] + 1
    #    start = end - label_len
    #    label_logits[i] = lsr_logits[i, start:end, :]
    token_loss = F.cross_entropy(
        lsr_logits.reshape(-1, lsr_logits.shape[-1]),
        #label_logits.reshape(-1, vocab_size),
        #label_attention_mask.view(-1),
        lsr_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction='none',
    )

    scores = token_loss.view(bsz, k, -1)
    #z = (label_attention_mask.view(bsz, k, -1) > -1).sum(dim=-1)
    z = (lsr_labels.view(bsz, k, -1) > -1).sum(dim=-1)
    lm_perplexity = scores.sum(dim=-1) / z  # negative if lower is better, otherwise positive
    lm_likelihood = torch.softmax(lm_perplexity / gamma, dim=-1)
    return lm_likelihood, lm_perplexity, lsr_input_ids

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

def compute_tags_likelihood(samples, gamma=5e-4, desc="tags"):
    bsz, k = np.array(samples[desc]).shape
    tags_scores = samples[f"top_scores_{desc}"].reshape((bsz, k)).to(device)
    tags_likelihood = torch.softmax(tags_scores / gamma, dim=-1)
    return tags_likelihood, tags_scores

def compute_loss(samples, labels, table_name=None, desc="tags"):
    tags_likelihood, tags_scores = compute_tags_likelihood(samples)
    with torch.no_grad():
        llm_likelihood, llm_perplexity, lsr_input_ids = compute_llm_likelihood(samples, labels)
    if table_name:
        table_data = {
            "Tags likelihood": tags_likelihood[0],
            "Tags scores": tags_scores[0],
            "LM likelihood": llm_likelihood[0],
            "LM perplexity": llm_perplexity[0],
        }
        indices = llm_perplexity[0].argsort(descending=True).detach().cpu().numpy()
        for k, v in table_data.items():
            table_data[k] = v.detach().to(dtype=torch.float16).cpu().numpy()[indices]
        table_data[desc] = np.array(samples[desc][0])[indices]
        table_data["Prompt"] = np.array([
            tokenizer.decode(lsr_input_ids[i]) for i in range(lsr_input_ids.shape[0])
        ])[indices]
        table = wandb.Table(data=pd.DataFrame(table_data))
        wandb.log({table_name: table})
    kl_penalty = F.kl_div(
        tags_likelihood.log(), llm_likelihood.log(),
        reduction="batchmean", log_target=True
    )
    return kl_penalty

def forward(images):
    print("Entering forward pass")
    samples = lens(
        images,
        return_tags=True,
        return_attributes=False,
        return_intensive_captions=False,
        return_prompt=True
    )
    print(torch.cuda.mem_get_info()[0] / 1e9)
    print("Completed forward pass")
    return samples

def main(num_epochs=5000, lr=1e-5, batch_size=1, train_size=1000, val_size=1000):
    wandb.init(project="lens-training-coco-dataset")
    save_path = "trained_model_attributes.pt"
    question = ["What is this image about?" for i in range(batch_size)]
    ds_name = "cifar10"
    train_ds_raw = load_dataset(ds_name, split="train", streaming=False)
    train_ds = LensDataset(train_ds_raw, processor)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    print("Created train loader")
    val_ds_raw = load_dataset(ds_name, split="test", streaming=False)
    val_ds = LensDataset(val_ds_raw, processor)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    print("Created val loader")
    optimizer = torch.optim.Adam(lens.clip_model.parameters(), lr=lr)
    print("Before prepare")
    #lens.clip_model, optimizer, train_dataloader = accelerator.prepare(
    #    lens.clip_model, optimizer, train_dataloader
    #)
    lens.clip_model = accelerator.prepare(lens.clip_model)
    print("After prepare")

    for epoch in range(num_epochs):
        #Compute train loss
        train_loss_epoch = 0
        total, correct = 0, 0
        for i, (images, labels) in enumerate(train_dataloader):
            if i >= (train_size // batch_size):
                continue
            samples = forward(images)
            train_loss = compute_loss(samples, labels, f"Epoch {epoch}: train")
            wandb.log({"train_loss": train_loss.item()})
            #accelator.backward(train_loss)
            #accelerator.log({"train_loss": train_loss.item()})
            train_loss_epoch += train_loss.item()
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                total += len(labels)
                correct += compute_class_acc(samples["prompts"][0], labels[0], llm_model, tokenizer)
            print(f"Finished batch {i}")
        wandb.log({"train_loss_epoch": train_loss_epoch / (train_size // batch_size)})
        wandb.log({"train_acc": correct / total})
        #accelerator.log({"train_loss_epoch": train_loss_epoch / (train_size // batch_size)})
        #Compute val loss
        val_loss_epoch = 0
        total, correct = 0, 0
        for i, (images, labels) in enumerate(val_dataloader):
            if i >= (val_size // batch_size):
                continue
            with torch.no_grad():
                samples = forward(images)
                val_loss = compute_loss(samples, labels, f"Epoch {epoch}: val").item()
                wandb.log({"val_loss": val_loss})
                #accelerator.log({"val_loss": val_loss})
                val_loss_epoch += val_loss
                total += len(labels)
                correct += compute_class_acc(samples["prompts"][0], labels[0], llm_model, tokenizer)
        wandb.log({"val_loss_epoch": val_loss_epoch / (val_size // batch_size)})
        wandb.log({"val_acc": correct / total})
        #accelerator.log({"val_loss_epoch": val_loss_epoch / (val_size // batch_size)})

if __name__ == "__main__":
    parser = ArgumentParser(description='Train',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    main()
