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
import pandas as pd
from evaluate import load

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lens_model = Lens()
processor = LensProcessor()
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", truncation_side='left', padding=True)
# llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
perplexity = load("perplexity", module_type="metric")

def compute_llm_likelihood(samples, labels, desc):
    batch_size, num_descs = np.array(samples[desc]).shape
    #Encode prompts and groundtruth answers
    input_texts = []
    for i in range(batch_size):
        for j in range(num_descs):
            prompt = create_prompt_sample(
                samples, i, desc_idx=j, mode=f"{desc}_only_single",
                question_prompt=samples["questions"][i]
            )
            input_texts.append(f"{prompt} {labels[i]}")
    results = perplexity.compute(
        model_id='gpt2', predictions=input_texts
    )
    perplexities = torch.tensor(results["perplexities"]).reshape((batch_size, num_descs))
    return perplexities.to(device, dtype=torch.float64)
    #Get logits for groundtruth sequence when conditioned on each prompt
    # outputs = llm_model(
    #     input_ids=prompt_encodings["input_ids"],
    #     attention_mask=prompt_encodings["attention_mask"],
    #     labels=label_encodings["input_ids"]
    # )
    # #Compute logprobs based on logits
    # _, seq_length, vocab_size = outputs.logits.shape
    # logits = outputs.logits.reshape((batch_size, num_descs, seq_length, vocab_size))
    # logprobs = logits.log_softmax(dim=-1)
    # #Return perplexity rather than likelihood to avoid underflow
    # token_ids = label_encodings["input_ids"].reshape((batch_size, num_descs, seq_length, 1))
    # masked_logprobs = logprobs.gather(dim=-1, index=token_ids).squeeze()
    # perplexity = masked_logprobs.mean(dim=-1).exp()
    # return perplexity.to(device)

def compute_desc_likelihood(samples, desc):
    scores = samples[f"top_scores_{desc}"].squeeze()
    return scores.to(device, dtype=torch.float64)

def compute_loss(samples, labels, desc, display=False):
    desc_likelihood = compute_desc_likelihood(samples, desc)
    desc_likelihood_soft = desc_likelihood.softmax(dim=-1)
    llm_likelihood = compute_llm_likelihood(samples, labels, desc)
    llm_likelihood_soft = llm_likelihood.softmax(dim=-1)
    if display:
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
        wandb.log({"Likelihoods": table})
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

def train(descs, num_epochs=50000, lr=1e-5, batch_size=8, train_size=8, val_size=8, early=5):
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
        best_train_loss, best_i = float('inf'), 0
        for i, batch in enumerate(train_dataloader):
            if i >= (train_size // batch_size) or (best_i - i) >= early:
                continue
            optimizer.zero_grad()
            samples = forward(batch, question, descs)
            train_loss = 0
            for desc in descs:
                kl_penalty = compute_loss(
                    samples, batch['caption'], desc, display=True
                )
                #wandb.log({f"train_kl_penalty_{desc}": kl_penalty})
                train_loss += kl_penalty
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_i = i
            wandb.log({"train_loss": train_loss})
            train_loss.backward()
            optimizer.step()
            torch.save(lens_model.state_dict(), save_path)

        # Compute train accuracy
        #for i, batch in enumerate(train_dataloader):
            #if i > (train_size // batch_size):
            #    continue
            #samples = forward(batch, question, descs)
            #train_acc = compute_accuracy(batch, samples)
            #wandb.log({"train_acc": train_acc})

        #Compute val loss
        for i, batch in enumerate(val_dataloader):
            if i >= (val_size // batch_size):
                continue
            val_loss = 0
            samples = forward(batch, question, descs)
            for j, desc in enumerate(descs):
                kl_penalty = compute_loss(samples, batch['caption'], desc)
                #wandb.log({f"val_kl_penalty_{desc}": kl_penalty})
                val_loss += kl_penalty
            wandb.log({"val_loss": val_loss})

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
