import open_clip
from transformers import CLIPModel, CLIPProcessor, AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from datasets import load_dataset
from utils import create_dataloader
from model import Lens, LensProcessor
import time
import torch
from vllm import LLM, SamplingParams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#llm_model = LLM("microsoft/phi-2")
#prompts = ["Write a sentence about the moon." ]
#sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
#outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
#for output in outputs:
#    import pdb; pdb.set_trace()
#    prompt = output.prompt
#    generated_text = output.outputs[0].text
#    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
#clip_name = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
#accelerator = Accelerator()
#model = open_clip.create_model_and_transforms(model_name)[0]
#model = CLIPModel.from_pretrained(clip_name)
#start = time.time()
lens_model = Lens()
#print(f"Minutes elapsed: {(time.time() - start) / 60}")
#processor = LensProcessor()
#print(f"Memory left: {torch.cuda.mem_get_info()[0] / 1e9}")
#llm_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True).to(device)
#tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
#batch_size = 8
#lr = 1e-3
#train_ds = load_dataset("cifar10", split="train", streaming=True)
#train_dataloader = create_dataloader(train_ds, batch_size=batch_size)
#optimizer = torch.optim.Adam(lens_model.clip_model.parameters(), lr=lr)
#lens_model.clip_model, optimizer, train_dataloader = accelerator.prepare(lens_model.clip_model, optimizer, train_dataloader)
#print("Finished successfully")
#print(f"Memory left: {torch.cuda.mem_get_info()[0] / 1e9}")
#llm_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True, device_map="auto")#.to(device) #Consumes 10+GB of memory!
#print(f"Memory left: {torch.cuda.mem_get_info()[0] / 1e9}")
#print("Initialized LLM model")
