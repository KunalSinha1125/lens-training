from transformers import AutoModelForCausalLM, AutoTokenizer
from model import Lens, LensDataset, LensProcessor
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"You are using device {device}")
lens = Lens()
processor = LensProcessor()
llm_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def compute_class_acc(prompts, labels):
    tokenizer.pad_token_id = tokenizer.eos_token_id
    input_ids = tokenizer(prompts, return_tensors="pt").input_ids.to(device)
    
    #outputs = llm_model.generate(input_ids, max_length=300)
    #outputs = outputs[0][input_ids.shape[-1]:]
    #pred = tokenizer.decode(outputs)
    #return (pred == labels[0])

def evaluate_pipeline():
    ds_name = "cifar10"
    batch_size = 1
    ds_raw = load_dataset(ds_name, split="train", streaming=False)
    ds = LensDataset(ds_raw, processor)
    dataloader = DataLoader(ds, batch_size=batch_size)
    correct, total = 0, 0
    for i, (images, labels) in enumerate(dataloader):
        if i > 10:
            continue
        with torch.no_grad():
            samples = lens(images, return_tags=True, return_prompt=True)
        print(samples["tags"])
        total += images.shape[0]
        correct += compute_class_acc(samples["prompts"], labels)

def test_prompts():
    #tags = ['777 200 aircraft', 'erj 135 aircraft', 'Chengdu j 10', 'Tupolev sb', 'Mil mi 24', 'bae 146 200 aircraft', 'Mahlab', 'a340 300 aircraft', 'a330 300 aircraft', 'aileron', 'Daewoo magnus', 'demonstrator', 'pa 28 aircraft', 'thumb', 'Khene', 'bat', 'Hongdu l 15', 'Kutia', 'Polo', 'Ogokbap']
    #tags = ['calf', 'yoke', 'Elder', 'Mule', 'Bouvier des ardennes', 'Sarangi', 'Kabusecha', 'Spur', 'Hongdu l 15', 'trinket', 'Kurri', 'Khene', 'bear', 'Merlin', 'bolotie', 'Sapphire', 'Oriental longhair', 'Bay', 'Auburn 851', 'Slider']
    tags = [['plane', '777 200 aircraft', '727 200 aircraft', 'erj 135 aircraft', 'Boeing 717', 'md 11 aircraft', '767 300 aircraft', 'Zha cai', 'Chengdu j 10', 'Tgv', 'Boeing 2707', 'airline', 'Xian h 6', 'crj 200 aircraft', 'prop', 'Trijet', '737 500 aircraft', 'Tteokguk', 'tu 154 aircraft', 'tu 134 aircraft']]
    all_prompts = [
            f"Instruct: you are given the image tags {tags}. Based on this information, output a word that describes the image. \nOutput:"
    ]
    for prompts in all_prompts:
        compute_class_acc(prompts, ["airplane"])

evaluate_pipeline()
