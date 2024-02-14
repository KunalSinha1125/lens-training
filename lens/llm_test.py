from model import Lens, LensDataset, LensProcessor
import requests
from PIL import Image
from scipy.special import rel_entr
from transformers import Trainer, TrainingArguments, CLIPProcessor, CLIPModel, GPT2LMHeadModel, GPT2TokenizerFast#AutoModelForCausalLM, AutoTokenizer
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
print(f"You are using device {device}")
llm_model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float16).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
#perplexity = load("perplexity", module_type="metric")
IGNORE_INDEX = -100

def compute_llm_likelihood(samples, labels, gamma=0.1, desc="tags"):
    bsz, k = np.array(samples[desc]).shape
    prompts = []
    k=1
    for i in range(bsz):
        for j in range(k):
            prompt = create_prompt_sample(
                samples, i, desc_idx=j, mode=f"{desc}_only_loop",
                question_prompt=samples["questions"][i]
            )
            prompts.append(prompt)

    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True)
    #tokenizer.padding_side = "right"
    label_tokens = tokenizer(labels, return_tensors="pt", padding=True)
    reader_tok, reader_mask = prompt_tokens.input_ids.to(device), prompt_tokens.attention_mask.to(device)
    answer_tok, answer_mask = label_tokens.input_ids.to(device), label_tokens.attention_mask.to(device)

    repeat_answer_tok = torch.repeat_interleave(answer_tok[:, None], k, dim=1).view(-1, answer_tok.shape[-1])
    repeat_answer_mask = torch.repeat_interleave(answer_mask[:, None], k, dim=1).view(-1, answer_mask.shape[-1])
    reader_tok = reader_tok.reshape(-1, reader_tok.shape[-1])
    reader_mask = reader_mask.reshape(-1, reader_mask.shape[-1])

    lsr_input_ids = torch.cat((reader_tok, repeat_answer_tok), dim=-1)
    lsr_attention_mask = torch.cat((reader_mask, repeat_answer_mask), dim=-1)
    with torch.autocast("cuda", dtype=torch.float16):
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
    lm_perplexity = scores.sum(dim=-1) / z  # negative if lower is better, otherwise positive
    lm_likelihood = torch.softmax(lm_perplexity / gamma, dim=-1)
    return lm_likelihood, lm_perplexity

def main(tags):
    train_ds = load_dataset("cifar10", split="train", streaming=True)
    train_dataloader = create_dataloader(train_ds, batch_size=1)
    question = ["What is this image about?"]
    samples = {
        "tags": tags,
        "question": question
    }
    for i, batch in enumerate(train_dataloader):
        if i >= 1:
            continue
        print(compute_llm_likelihood(samples, question))

if __name__ == "__main__":
    parser = ArgumentParser(description='Train',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    tags = args.tags if args.tags else []
    main(tags)
