from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

def setup_lora(model_name="Qwen/Qwen3-0.6B"):
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return model

if __name__ == "__main__":
    model = setup_lora()
    print("LoRA model ready.") 