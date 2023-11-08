# LoRA = Low rank adaptation parameters --> PEFT = parameter efficient fine tuning
# h(x) = Wx + BAx  where B, A \in R^n --> B@A \in R^n**2

# from datasets import load_dataset, DatasetDict, Dataset

# from transformers import (
#     AutoTokenizer, 
#     AutoConfig, 
#     AutoModelFor, 
#     DataCollatorWithPadding,
#     TrainingArguments,
#     Trainer)

# from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
# import evaluate
# import torch 
# import numpy as np 

# model_checkpoint = 'distillbert-base-uncased'


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = 'mps'

model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-34B", device_map="auto", torch_dtype=torch.float32, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34B", trust_remote_code=True)
inputs = tokenizer("There's a place where time stands still. A place of breath taking wonder, but also", return_tensors="pt")
max_length = 256  

outputs = model.generate(
    inputs.input_ids.to(device),
    max_length=max_length,
    eos_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
