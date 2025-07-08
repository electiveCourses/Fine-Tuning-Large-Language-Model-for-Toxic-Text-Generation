from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig
from datasets import load_dataset, Dataset
import torch

def main():
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # Загрузка датасета
    dataset = load_dataset("parquet", data_files={"train": "data/processed/cleaned.parquet"})

    # Получаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Функция токенизации
    def tokenize_function(batch):
        tokens = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        # Гарантируем двумерность labels
        if isinstance(tokens["input_ids"][0], list):
            tokens["labels"] = [ids.copy() for ids in tokens["input_ids"]]
        else:
            tokens["labels"] = [tokens["input_ids"].copy()]
        return tokens

    # Токенизация всего датасета
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Оставим только первые 200 примеров для быстрой тренировки
    small_train_dataset = tokenized_dataset["train"].select(range(200))

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        max_steps=50,
        logging_steps=5,
        save_steps=25,
        fp16=False,  # обязательно!
        bf16=False,  # тоже лучше явно отключить
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

if __name__ == "__main__":
    print("Torch device:", torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    main() 