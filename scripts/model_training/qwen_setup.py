from transformers import AutoModelForCausalLM, AutoTokenizer

def load_qwen_model(model_name="Qwen/Qwen3-0.6B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_qwen_model()
    prompt = "Пример токсичного комментария:"
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=100, temperature=0.7, top_k=50)
    print(tokenizer.decode(output[0], skip_special_tokens=True)) 