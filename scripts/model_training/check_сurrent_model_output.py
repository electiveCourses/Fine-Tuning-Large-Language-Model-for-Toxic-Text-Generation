from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

checkpoint_path = "./results/checkpoint-50"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

model = model.to(device)

prompt = "Как дела?"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))