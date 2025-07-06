from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def main():
    model_name = "Qwen/Qwen1.5-1.8B"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Загрузите датасет пар для DPO
    dataset = load_dataset("path/to/your/reward_model_pairs")  # заменить на ваш путь
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=None,  # добавить TrainingArguments
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        tokenizer=tokenizer,
        beta=0.1,
    )
    trainer.train()

if __name__ == "__main__":
    main() 