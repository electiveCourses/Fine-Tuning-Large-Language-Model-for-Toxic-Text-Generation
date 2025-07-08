from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

def generate_samples(model, tokenizer, prompt, n=100):
    samples = []
    for _ in range(n):
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(**inputs, max_length=100, temperature=0.7, top_k=50)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        samples.append(text)
    return samples

if __name__ == "__main__":
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompt = "Пример токсичного комментария:"
    samples = generate_samples(model, tokenizer, prompt, n=100)
    df = pd.DataFrame({"generated": samples})
    df.to_csv("data/processed/qwen_baseline_samples.csv", index=False)
    print("Saved baseline generations to data/processed/qwen_baseline_samples.csv") 