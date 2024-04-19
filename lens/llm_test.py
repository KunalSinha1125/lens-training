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
print(f"You are using device {device}")
#model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", attn_implementation="flash_attention_2").to(device, dtype=torch.float16)
#tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
llm_model = GPT2LMHeadModel.from_pretrained("gpt2-xl", torch_dtype=torch.float16).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")
#perplexity = load("perplexity", module_type="metric")
IGNORE_INDEX = -100

def compute_llm_likelihood(samples, labels, gamma=0.1, desc="tags"):
    bsz, k = np.array(samples[desc]).shape
    #prompts = []
    inputs = []
    for i in range(bsz):
        for j in range(k):
            prompt = create_prompt_sample(
                samples, i, desc_idx=j, mode=f"{desc}_only_test",
                question_prompt="placeholder"
            )
            inputs.append(f"{prompt} {labels[i]}")
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    lsr_tokens = tokenizer(inputs, return_tensors="pt", padding=True)
    lsr_input_ids = lsr_tokens.input_ids.to(device)
    lsr_attention_mask = lsr_tokens.attention_mask.to(device)
    #prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True)
    #label_tokens = tokenizer(labels, return_tensors="pt", padding=True)
    #reader_tok, reader_mask = prompt_tokens.input_ids.to(device), prompt_tokens.attention_mask.to(device)
    #reader_tok = torch.cat([torch.tensor([[tokenizer.pad_token_id, tokenizer.pad_token_id]]).to(device), reader_tok], dim=1)
    #reader_mask = torch.cat([torch.tensor([[0, 0]]).to(device), reader_mask], dim=1)
    #import pdb; pdb.set_trace()
    #answer_tok, answer_mask = label_tokens.input_ids.to(device), label_tokens.attention_mask.to(device)

    #repeat_answer_tok = torch.repeat_interleave(answer_tok[:, None], k, dim=1).view(-1, answer_tok.shape[-1])
    #repeat_answer_mask = torch.repeat_interleave(answer_mask[:, None], k, dim=1).view(-1, answer_mask.shape[-1])
    #reader_tok = reader_tok.reshape(-1, reader_tok.shape[-1])
    #reader_mask = reader_mask.reshape(-1, reader_mask.shape[-1])

    #lsr_input_ids = torch.cat((reader_tok, repeat_answer_tok), dim=-1)
    for i in range(lsr_input_ids.shape[0]):
        print(tokenizer.decode(lsr_input_ids[i]))
    #lsr_attention_mask = torch.cat((reader_mask, repeat_answer_mask), dim=-1)
    with torch.autocast("cuda", dtype=torch.float16):
        with torch.no_grad():
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
    import pdb; pdb.set_trace()
    return lm_likelihood, lm_perplexity

def main(tags):
    samples = { "tags": [tags] }
    labels = ["airplane"]
    _, lm_perplexity = compute_llm_likelihood(samples, labels)
    indices = lm_perplexity[0].argsort(descending=True)
    print(lm_perplexity[0][indices])
    print(np.array(tags)[indices.cpu().numpy()])

if __name__ == "__main__":
    parser = ArgumentParser(description='Train',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tags',
                        nargs='+',
                        help='What tags to test perplexity on?')
    args = parser.parse_args()
    tags = []
    if args.tags:
        for t in args.tags:
            if t.endswith("txt"):
                with open(t, "r") as f:
                    text = [line.strip() for line in f.readlines()]
                    tags += text
            else:
                tags.append(t)
    main(tags)
