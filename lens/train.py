from model import Lens, LensDataset, LensProcessor
import requests
from PIL import Image
from scipy.special import rel_entr
from transformers import Trainer, TrainingArguments, CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
from utils import create_prompt_sample, create_dataloader, create_sampler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
import wandb
import matplotlib.pyplot
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lens_model = Lens()
processor = LensProcessor()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", truncation_side='left', padding=True)
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def logits_to_probabilities(logits, token_ids):
    batch_size, num_sequences, seq_length, vocab_size = logits.shape
    logprobs = logits.log_softmax(dim=-1)
    masked_logprobs = logprobs.gather(dim=-1, index=token_ids.unsqueeze(-1))
    log_likelihood = masked_logprobs.squeeze().sum(dim=-1)
    return torch.exp(log_likelihood)

def compute_llm_likelihood(samples, labels, desc):
    batch_size, num_descs = np.array(samples[desc]).shape
    #Encode prompts and groundtruth answers
    all_prompts, all_labels = [], []
    for i in range(batch_size):
        for j in range(num_descs):
            all_prompts.append(create_prompt_sample(
                samples, i, desc_idx=j, mode=f"{desc}_only_single",
                question_prompt=samples["questions"][i]
            ))
            all_labels.append(labels[i])
    prompt_encodings = tokenizer(all_prompts, return_tensors="pt", padding=True)
    label_encodings = tokenizer(all_labels, return_tensors="pt", padding=True)
    #Get logits for groundtruth sequence when conditioned on each prompt
    outputs = llm_model(
        input_ids=prompt_encodings["input_ids"],
        attention_mask=prompt_encodings["attention_mask"],
        labels=label_encodings["input_ids"]
    )
    #Compute likelihood based on logits
    _, seq_length, vocab_size = outputs.logits.shape
    logits = outputs.logits.reshape((batch_size, num_descs, seq_length, vocab_size))
    token_ids = label_encodings["input_ids"].reshape((batch_size, num_descs, seq_length))
    return logits_to_probabilities(logits, token_ids).to(device)

def compute_desc_likelihood(samples, desc):
    if desc == "intensive_captions":
        logits = samples[f"{desc}_logits"]
        batch_size, num_descs = np.array(samples[desc]).shape
        _, seq_length = logits.shape
        logits = logits.reshape((batch_size, num_descs, seq_length))
        log_likelihood = logits.sum(dim=-1)
        return torch.exp(log_likelihood)
    return samples[f"top_scores_{desc}"].squeeze().to(device)

def compute_loss(samples, labels, desc, alpha):
    desc_likelihood = compute_desc_likelihood(samples, desc)
    llm_likelihood = compute_llm_likelihood(samples, labels, desc)
    kl_penalty = F.kl_div(
        desc_likelihood.log_softmax(dim=-1), llm_likelihood.log_softmax(dim=-1),
        reduction="batchmean", log_target=True
    )
    return alpha * kl_penalty

def forward(batch, question, descs):
    inputs = processor(batch['image'], question)
    samples = lens_model(
        inputs, 
        return_tags=("tags" in descs),
        return_attributes=("attributes" in descs),
        return_prompt=True
    )
    return samples

def compute_accuracy(batch, samples):
    input_ids = tokenizer(samples["prompts"], return_tensors="pt").input_ids
    outputs = llm_model.generate(input_ids)
    predictions = tokenizer.decode(outputs[0])
    answers = batch["captions"]
    num_correct = torch.eq(predictions, answers).sum()
    return num_correct

def train(descs, num_epochs=5, lr=1e-5, batch_size=8, training_size=8, val_size=8, early=5):
    wandb.init(project="lens-training-coco-dataset")
    save_path = "trained_model_" + "_".join(descs) + ".pt"
    question = ["What is the image about" for i in range(batch_size)]
    train_ds = load_dataset("RIW/small-coco", split="train", streaming=True)
    train_dataloader = create_dataloader(train_ds, batch_size=batch_size)
    val_ds = load_dataset("RIW/small-coco", split="validation", streaming=True)
    val_dataloader = create_dataloader(val_ds, batch_size=batch_size)
    optimizer = torch.optim.Adam(lens_model.parameters(), lr=lr)
    torch.autograd.set_detect_anomaly(True)
    alphas = {
        "tags": 1,
        "attributes": 100
    }
    for epoch in range(num_epochs):
        train_loss_epoch = 0
        best_train_loss, best_i = float('inf'), 0
        for i, batch in enumerate(train_dataloader):
            if i > (training_size // batch_size) or (best_i - i) >= early:
                continue
            optimizer.zero_grad()
            samples = forward(batch, question, descs)
            train_loss = 0
            for j, desc in enumerate(descs):
                kl_penalty = compute_loss(samples, batch['caption'], desc, alphas[desc])
                wandb.log({f"train_kl_penalty_{desc}": kl_penalty})
                train_loss += kl_penalty
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_i = i
            train_loss_epoch += train_loss
            wandb.log({"train_loss": train_loss})
            train_loss.backward()
            optimizer.step()
            torch.save(lens_model.state_dict(), save_path)

        train_acc_epoch = 0
        for i, batch in enumerate(train_dataloader):
            if i > (training_size // batch_size):
                continue
            samples = forward(batch, question, descs)
            num_correct = compute_accuracy(batch, samples)
            train_acc_epoch += num_correct
            wandb.log({"train_acc": num_correct / len(samples["prompts"])})

        val_loss_epoch = 0
        for i, batch in enumerate(val_dataloader):
            if i > (val_size // batch_size):
                continue
            val_loss = 0
            samples = forward(batch, question, descs)
            for j, desc in enumerate(descs):
                kl_penalty = compute_loss(samples, batch['caption'], desc, alphas[desc])
                wandb.log({f"val_kl_penalty_{desc}": kl_penalty})
                val_loss += kl_penalty
            val_loss_epoch += val_loss
            wandb.log({"val_loss": val_loss})
        wandb.log({"train_loss_epoch": train_loss_epoch})
        wandb.log({"val_loss_epoch": val_loss_epoch})

if __name__ == "__main__":
    parser = ArgumentParser(description='Train',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--descriptions',
                        nargs='+',
                        help='Which descriptions to train on')
    args = parser.parse_args()
    descs = args.descriptions if args.descriptions else ["tags", "attributes"]
    train(descs)
