from model import Lens, LensDataset, LensProcessor
import requests
from PIL import Image
from scipy.special import rel_entr
from transformers import Trainer, TrainingArguments, CLIPProcessor, CLIPModel, GPT2LMHeadModel, GPT2TokenizerFast
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
from evaluate import load

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lens_model = Lens()
processor = LensProcessor()
llm_model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", padding=True)
IGNORE_INDEX = -100

def compute_llm_likelihood(samples, labels, gamma=1.0):
    bsz, k = np.array(samples["tags"]).shape
    all_prompts, all_labels = [], []
    for i in range(bsz):
        for j in range(k):
            prompt = create_prompt_sample(
                samples, i, desc_idx=j, mode=f"tags_only_single",
                question_prompt=samples["questions"][i]
            )
            all_prompts.append(prompt)
            all_labels.append(labels[i])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    prompt_tokens = tokenizer(all_prompts, return_tensors="pt", padding=True)
    label_tokens = tokenizer(all_labels, return_tensors="pt", padding=True)

    reader_tok, reader_mask = prompt_tokens.input_ids, prompt_tokens.attention_mask
    answer_tok, answer_mask = label_tokens.input_ids, label_tokens.attention_mask

    retrieved_tags = tokenizer(samples["tags"], return_tensors="pt", padding=True).input_ids
    repeat_answer_tok = torch.repeat_interleave(answer_tok[:, None], retrieved_tags, dim=1).view(-1, answer_tok.shape[-1])
    repeat_answer_mask = torch.repeat_interleave(answer_mask[:, None], retrieved_tags, dim=1).view(-1, answer_mask.shape[-1])
    reader_tok = reader_tok.reshape(-1, reader_tok.shape[-1])
    reader_mask = reader_mask.reshape(-1, reader_mask.shape[-1])

    lsr_input_ids = torch.cat((reader_tok, repeat_answer_tok), dim=-1)
    lsr_attention_mask = torch.cat((reader_mask, repeat_answer_mask), dim=-1)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.inference_mode():
            lsr_logits = llm_model(
                input_ids=lsr_input_ids[:, :-1],
                attention_mask=lsr_attention_mask[:, :-1],
                use_cache=False,
            ).logits

    # compute perplexity of question
    continuation_length = repeat_answer_tok.shape[-1]
    lsr_logits = lsr_logits[:, -continuation_length:]
    lsr_labels = repeat_answer_tok.masked_fill(repeat_answer_mask == 0, IGNORE_INDEX)
    token_loss = F.cross_entropy(
        lsr_logits.reshape(-1, lsr_logits.shape[-1]),
        lsr_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction='none',
    )

    scores = token_loss.view(bsz, k, -1)
    z = (lsr_labels.view(bsz, k, -1) > -1).sum(dim=-1)
    lm_perplexity = -scores.sum(dim=-1) / z  # lower is better
    lm_likelihood = torch.softmax(lm_perplexity / gamma, dim=-1)
    return lm_likelihood.to(device, dtype=torch.float64), lm_perplexity

def compute_tags_likelihood(samples, gamma=1.0):
    scores = samples[f"top_scores_tags"].squeeze()
    return torch.softmax(scores / gamma, dim=-1).to(device, dtype=torch.float64)

def compute_loss(samples, labels, plot=False):
    tags_likelihood = compute_tags_likelihood(samples)
    llm_likelihood, llm_perplexity = compute_llm_likelihood(samples, labels)
    if plot:
        table_data = {
            "LM likelihood": llm_likelihood[0],
            "LM perplexity": llm_perplexity[0],
        }
        for k, v in table_data.items():
            table_data[k] = v.detach().cpu().numpy()
        table = wandb.Table(data=pd.DataFrame(table_data))
        wandb.log({f"LM examples": table})
    kl_penalty = F.kl_div(
        tags_likelihood.log(), llm_likelihood.log(),
        reduction="batchmean", log_target=True
    )
    return kl_penalty

def forward(batch, question):
    inputs = processor(batch['image'], question)
    samples = lens_model(
        inputs, 
        return_tags=True,
        return_prompt=True
    )
    return samples

def train(num_epochs=50000, lr=1e-5, batch_size=8, train_size=8, val_size=64):
    wandb.init(project="lens-training-coco-dataset")
    save_path = "trained_model_tags.pt"
    question = ["What is the image about" for i in range(batch_size)]
    train_ds = load_dataset("RIW/small-coco", split="train", streaming=True)
    train_dataloader = create_dataloader(train_ds, batch_size=batch_size)
    val_ds = load_dataset("RIW/small-coco", split="validation", streaming=True)
    val_dataloader = create_dataloader(val_ds, batch_size=batch_size)
    optimizer = torch.optim.Adam(lens_model.parameters(), lr=lr)
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(num_epochs):
        #Compute train loss
        train_loss_epoch = 0
        for i, batch in enumerate(train_dataloader):
            if i >= (train_size // batch_size):
                continue
            optimizer.zero_grad()
            samples = forward(batch, question)
            train_loss = compute_loss(
                samples, batch['caption'], 
                plot = (epoch == 0) and (i == 0)    
            )
            wandb.log({"train_loss": train_loss})
            train_loss.backward()
            optimizer.step()
            torch.save(lens_model.state_dict(), save_path)
            train_loss_epoch += train_loss
        wandb.log({"train_loss_epoch": train_loss_epoch / (train_size // batch_size)})

        #Compute val loss
        val_loss_epoch = 0
        for i, batch in enumerate(val_dataloader):
            if i >= (val_size // batch_size):
                continue
            samples = forward(batch, question)
            val_loss = compute_loss(samples, batch['caption'])
            wandb.log({"val_loss": val_loss})
            val_loss_epoch += val_loss
        wandb.log({"val_loss_epoch": val_loss_epoch / (val_size // batch_size)})

if __name__ == "__main__":
    parser = ArgumentParser(description='Train',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    train()
