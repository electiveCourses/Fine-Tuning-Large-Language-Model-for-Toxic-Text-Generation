from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def main():
    model_name = "Qwen/Qwen1.5-0.5B"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
   
    dataset = load_dataset("data/processed/reward_model_pairs.parquet")  # заменить на ваш путь
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=None,  
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        tokenizer=tokenizer,
        beta=0.1,
    )
    trainer.train()

if __name__ == "__main__":
    main() 