from transformers import AutoModelForCausalLM, AutoTokenizer
from model import Lens, LensDataset, LensProcessor, CACHE_DIR
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IGNORE_INDEX = -100

def compute_class_acc(prompts, groundtruths, llm_model, tokenizer, all_classes, buffer_size=125):
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    prompt_tokens = tokenizer(prompts, return_tensors="pt", add_special_tokens=True, padding=True).to(device)
    reader_tok, reader_mask = prompt_tokens.input_ids, prompt_tokens.attention_mask
    batch_size, num_classes = len(prompts), min(len(all_classes), buffer_size)
    reader_tok = torch.repeat_interleave(reader_tok[:, None], num_classes, dim=1).view(-1, reader_tok.shape[-1])
    reader_mask = torch.repeat_interleave(reader_mask[:, None], num_classes, dim=1).view(-1, reader_mask.shape[-1]) 
    idx = 0
    cross_entropy = []
    while idx < len(all_classes):
        classes = all_classes[idx : idx + num_classes]
        tokenizer.padding_side = "right"
        label_tokens = tokenizer(classes, return_tensors="pt", add_special_tokens=True, padding=True).to(device)
        answer_tok, answer_mask = label_tokens.input_ids, label_tokens.attention_mask
        answer_tok = torch.repeat_interleave(answer_tok[None, :], batch_size, dim=0).view(-1, answer_tok.shape[-1])
        answer_mask = torch.repeat_interleave(answer_mask[None, :], batch_size, dim=0).view(-1, answer_mask.shape[-1])
        
        lsr_input_ids = torch.cat((reader_tok, answer_tok), dim=-1).to(device)
        lsr_attention_mask = torch.cat((reader_mask, answer_mask), dim=-1).to(device)
        with torch.autocast("cuda"):
            lsr_logits = llm_model(
                input_ids=lsr_input_ids[:, :-1],
                attention_mask=lsr_attention_mask[:, :-1],
                use_cache=False,
            ).logits
        continuation_length = answer_tok.shape[-1]
        lsr_logits = lsr_logits[:, -continuation_length:]
        lsr_labels = answer_tok.masked_fill(answer_mask == 0, IGNORE_INDEX).to(device)
        token_loss = F.cross_entropy(
            lsr_logits.reshape(-1, lsr_logits.shape[-1]),
            lsr_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction='none',
        ).reshape((batch_size, num_classes, -1))
        z = (lsr_labels.reshape((batch_size, num_classes, -1)) > -1).sum(dim=-1)
        cross_entropy.append(token_loss.sum(dim=-1) / z)
        idx += num_classes
    cross_entropy = torch.cat(cross_entropy, dim=-1)
    print("Cross entropy: ", cross_entropy)
    predictions = cross_entropy.argmin(dim=-1).cpu().numpy()
    correct = (np.array(all_classes)[predictions] == np.array(groundtruths)).sum()
    print("Correctness: ", correct)
    return correct
    #outputs = llm_model.generate(input_ids, max_length=300)
    #outputs = outputs[0][input_ids.shape[-1]:]
    #pred = tokenizer.decode(outputs)
    #return (pred == labels[0])

def compute_vqa_acc(prompts, groundtruths, llm_model, tokenizer, question_types=None, correct_by_type={}):
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    max_new_tokens = max([len(prompt) for prompt in prompts]) + 100
    output = llm_model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    generations = tokenizer.batch_decode(output)
    preds = np.array([gen.split("\"")[-2].lower().replace(".", "") for gen in generations])
    correct = (preds == np.array(groundtruths)).sum()
    if question_types:
        for i, typ in enumerate(question_types):
            if typ not in correct_by_type:
                correct_by_type[typ] = [0, 0]
            correct_by_type[typ][0] += (preds[i] == groundtruths[i])
            correct_by_type[typ][1] += 1
    print("Prompts: ", prompts)
    print("Predictions: ", preds)
    print("Groundtruth: ", groundtruths)
    print("Correctness: ", correct)
    return correct

def evaluate_pipeline(dataloader, lens, processor, llm_model, tokenizer, 
                      data_size=1000, batch_size=8, task="vqa"):
    correct, total = 0, 0
    correct_by_type = {}
    for i, (clip_image, blip_image, blip_input_ids, questions, question_types, labels) in enumerate(dataloader):
        if i > data_size // batch_size:
            continue
        with torch.no_grad():
            samples = lens(
                clip_image, blip_image, blip_input_ids,
                return_intensive_captions=True, return_prompt=True, 
                questions=questions
            )
        total += batch_size
        if task == "vqa":
            correct += compute_vqa_acc(samples["prompts"], labels, llm_model, tokenizer, question_types, correct_by_type)
        else:
            correct += compute_class_acc(samples["prompts"][0], labels[0], llm_model, tokenizer)
        print(correct, total)
        print(correct_by_type)
    print(f"Final accuracy: {correct/total}")

def test_prompts(llm_model, tokenizer):
    #tags = ['777 200 aircraft', 'erj 135 aircraft', 'Chengdu j 10', 'Tupolev sb', 'Mil mi 24', 'bae 146 200 aircraft', 'Mahlab', 'a340 300 aircraft', 'a330 300 aircraft', 'aileron', 'Daewoo magnus', 'demonstrator', 'pa 28 aircraft', 'thumb', 'Khene', 'bat', 'Hongdu l 15', 'Kutia', 'Polo', 'Ogokbap']
    #tags = ['calf', 'yoke', 'Elder', 'Mule', 'Bouvier des ardennes', 'Sarangi', 'Kabusecha', 'Spur', 'Hongdu l 15', 'trinket', 'Kurri', 'Khene', 'bear', 'Merlin', 'bolotie', 'Sapphire', 'Oriental longhair', 'Bay', 'Auburn 851', 'Slider']
    tags = [['plane', '777 200 aircraft', '727 200 aircraft', 'erj 135 aircraft', 'Boeing 717', 'md 11 aircraft', '767 300 aircraft', 'Zha cai', 'Chengdu j 10', 'Tgv', 'Boeing 2707', 'airline', 'Xian h 6', 'crj 200 aircraft', 'prop', 'Trijet', '737 500 aircraft', 'Tteokguk', 'tu 154 aircraft', 'tu 134 aircraft']]
    all_prompts = [
            f"Instruct: you are given the image tag {tags}. Based on this information, output a word that describes the image. \nOutput: "
    ]
    for prompt in all_prompts:
        compute_class_acc(prompt, "airplane", llm_model, tokenizer)

def interactive_test(llm_model, tokenizer, vqa=True):
    while True:
        prompt = input("Specify prompt: ")
        answer = input("Specify answer: ")
        if vqa:
            compute_class_acc([prompt], [answer], llm_model, tokenizer, [answer])
        #all_classes = ["hat", "water", "shirt", "bottle", "chip", "bowl", "computer", "wheat", "bird"]
        #compute_class_acc([prompt, prompt], [answer, answer], llm_model, tokenizer, all_classes)

def generate_test(llm_model, tokenizer):
    while True:
        prompt = input("Specify prompt: ")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = llm_model.generate(**model_inputs, max_new_tokens=1000)
        print(tokenizer.decode(output[0]))

def main():
    lens = Lens()
    processor = LensProcessor()
    ds_name = "HuggingFaceM4/VQAv2"
    ds_raw = load_dataset(ds_name, split="train", streaming=True, cache_dir=CACHE_DIR)
    ds_raw = ds_raw.shuffle(seed=0, buffer_size=10000)
    ds = LensDataset(ds_raw, processor, ds_name)
    data_size, batch_size = 40000, 8
    dataloader = DataLoader(ds, batch_size=batch_size)
    llm_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", trust_remote_code=True,
        cache_dir=CACHE_DIR).to(device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True, cache_dir=CACHE_DIR)
    #generate_test(llm_model, tokenizer)
    #interactive_test(llm_model, tokenizer)
    evaluate_pipeline(dataloader, lens, processor, llm_model, tokenizer, data_size, batch_size)

main()
