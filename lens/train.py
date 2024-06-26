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

def compute_cross_entropy(prompt_ids, label_ids, ignore_index=-100):
    input_ids = torch.cat([prompt_ids, label_ids], dim=-1)
    num_seqs, seq_len = input_ids.shape
    target_ids = input_ids.clone()
    #Ignore prompt and padding tokens in loss calculation
    target_ids[:, :prompt_ids.size(-1)] = ignore_index
    target_ids[target_ids == tokenizer.pad_token_id] = ignore_index
    logits = llm_model(input_ids, labels=target_ids).logits
    loss = F.cross_entropy(
        logits.reshape((num_seqs * seq_len, -1)),
        target_ids.reshape((num_seqs * seq_len)),
        reduction="none",
        ignore_index=ignore_index
    )
    return loss.reshape((num_seqs, seq_len)).mean(dim=-1)

def compute_llm_likelihood(samples, labels, desc):
    batch_size, num_descs = np.array(samples[desc]).shape
    #Encode prompts and groundtruth answers
    #losses = torch.zeros((batch_size, num_descs))
    all_prompts, all_labels = [], []
    for i in range(batch_size):
        for j in range(num_descs):
            prompt = create_prompt_sample(
                samples, i, desc_idx=j, mode=f"{desc}_only_single",
                question_prompt=samples["questions"][i]
            )
            all_prompts.append(prompt)
            all_labels.append(labels[i])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    prompt_ids = tokenizer(all_prompts, return_tensors="pt", padding=True).input_ids
    label_ids = tokenizer(all_labels, return_tensors="pt", padding=True).input_ids
    loss = compute_cross_entropy(prompt_ids, label_ids).reshape((batch_size, num_descs))
    return loss.to(device, dtype=torch.float64)

def compute_desc_likelihood(samples, desc):
    scores = samples[f"top_scores_{desc}"].squeeze()
    return scores.to(device, dtype=torch.float64)

def compute_loss(samples, labels, desc, plot_name=None):
    desc_likelihood = compute_desc_likelihood(samples, desc)
    desc_likelihood_soft = desc_likelihood.softmax(dim=-1)
    llm_likelihood = compute_llm_likelihood(samples, labels, desc)
    llm_likelihood_soft = llm_likelihood.softmax(dim=-1)
    if plot_name:
        table_data = {
            "L_D": desc_likelihood[0],
            "L_LM": llm_likelihood[0],
            "L_D soft": desc_likelihood_soft[0],
            "L_LM soft": llm_likelihood_soft[0],
        }
        for k, v in table_data.items():
            table_data[k] = v.detach().cpu().numpy()
        table_data["tags"] = samples["tags"][0]
        table = wandb.Table(data=pd.DataFrame(table_data))
        #wandb.log({f"{plot_name}_table": table})
        plot = wandb.plot.scatter(
            table, "L_D soft", "L_LM soft", title=f"{plot_name}"
        )
        wandb.log({plot_name: plot})
    kl_penalty = F.kl_div(
        desc_likelihood_soft.log(), llm_likelihood_soft.log(),
        reduction="batchmean", log_target=True
    )
    return kl_penalty

def forward(batch, question, descs):
    inputs = processor(batch['image'], question)
    samples = lens_model(
        inputs, 
        return_tags=("tags" in descs),
        return_attributes=("attributes" in descs),
        return_prompt=True
    )
    return samples

# def compute_accuracy(batch, samples):
#     input_ids = tokenizer(samples["prompts"], return_tensors="pt", padding=True).input_ids
#     outputs = llm_model.generate(input_ids)
#     predictions = np.array([
#         pred.replace("</pad>", "").replace("</s", "").strip()
#         for pred in tokenizer.batch_decode(outputs)
#     ])
#     answers = np.array(batch["captions"])
#     acc = (predictions == answers).mean()
#     return acc

def train(descs, num_epochs=50000, lr=1e-5, batch_size=8, train_size=8, val_size=400):
    wandb.init(project="lens-training-coco-dataset")
    save_path = "trained_model_" + "_".join(descs) + ".pt"
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
            samples = forward(batch, question, descs)
            train_loss = 0
            for desc in descs:
                kl_penalty = compute_loss(samples, batch['caption'], desc)
                train_loss += kl_penalty
            #wandb.log({"train_loss": train_loss})
            train_loss.backward()
            optimizer.step()
            torch.save(lens_model.state_dict(), save_path)
            train_loss_epoch += train_loss.item()
        wandb.log({"train_loss_epoch": train_loss_epoch / (train_size // batch_size)})

        # Compute train accuracy
        #for i, batch in enumerate(train_dataloader):
            #if i > (train_size // batch_size):
            #    continue
            #samples = forward(batch, question, descs)
            #train_acc = compute_accuracy(batch, samples)
            #wandb.log({"train_acc": train_acc})

        #Compute val loss
        val_loss_epoch = 0
        for i, batch in enumerate(val_dataloader):
            if i >= (val_size // batch_size):
                continue
            samples = forward(batch, question, descs)
            val_loss = 0
            for desc in descs:
                plot_name = "train likelihoods" if i == 0 and epoch == 0 else None
                kl_penalty = compute_loss(samples, batch['caption'], desc, plot_name=plot_name).item()
                val_loss += kl_penalty
            #wandb.log({"val_loss": val_loss})
            val_loss_epoch += val_loss
        wandb.log({"val_loss_epoch": val_loss_epoch / (val_size // batch_size)})

        # Compute val accuracy
        #for i, batch in enumerate(val_dataloader):
            #if i > (val_size // batch_size):
                #continue
            #samples = forward(batch, question, descs)
            #val_acc = compute_accuracy(batch, samples)
            #wandb.log({"val_acc": val_acc})

if __name__ == "__main__":
    parser = ArgumentParser(description='Train',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--descriptions',
                        nargs='+',
                        help='Which descriptions to train on')
    args = parser.parse_args()
    descs = args.descriptions if args.descriptions else ["tags", "attributes"]
    train(descs)
