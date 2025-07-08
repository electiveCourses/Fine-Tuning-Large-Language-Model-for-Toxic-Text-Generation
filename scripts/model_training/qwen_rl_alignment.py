from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer


def main():
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
   
    dataset = load_dataset("parquet", data_files={"train": "data/processed/reward_model_pairs.parquet"})
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
