#!/usr/bin/env python3
"""
Hybrid QLoRA + DoRA fine-tuning for Qwen
L = L_QLoRA + lambda_dora * L_DoRA
"""

import os
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig
from datasets import load_dataset

def main():
    model_name = "Qwen/Qwen3-0.6B"
    data_path = "data/processed/cleaned.parquet"
    output_dir = "results/hybrid_lora_dora"
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    lambda_dora = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = get_peft_model(model, lora_config)
    hidden_size = model.config.hidden_size
    vocab_size = model.config.vocab_size

    class DoRAHead(nn.Module):
        def __init__(self, hidden_size, out_size):
            super().__init__()
            self.linear = nn.Linear(hidden_size, out_size)
        def forward(self, hidden_states):
            return self.linear(hidden_states)

    class HybridQwenLoraDoRA(nn.Module):
        def __init__(self, lora_model, dora_head, lambda_dora):
            super().__init__()
            self.lora_model = lora_model
            self.dora_head = dora_head
            self.lambda_dora = lambda_dora
            self.dora_loss_fct = nn.MSELoss()
        def forward(self, input_ids, attention_mask=None, labels=None):
            outputs = self.lora_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                return_dict=True
            )
            loss = outputs.loss
            last_hidden = outputs.hidden_states[-1]
            if labels is not None:
                one_hot = torch.nn.functional.one_hot(labels, num_classes=self.lora_model.config.vocab_size).float()
                dora_logits = self.dora_head(last_hidden)
                dora_loss = self.dora_loss_fct(dora_logits, one_hot)
                total_loss = loss + self.lambda_dora * dora_loss
            else:
                dora_loss = torch.tensor(0.0, device=loss.device)
                total_loss = loss
            return {"loss": total_loss, "lora_loss": loss, "dora_loss": dora_loss}

    dora_head = DoRAHead(hidden_size, vocab_size)
    hybrid_model = HybridQwenLoraDoRA(model, dora_head, lambda_dora)
    hybrid_model.to(device)

    # Загрузка датасета
    dataset = load_dataset("parquet", data_files={"train": data_path})

    def tokenize_function(batch):
        tokens = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        if isinstance(tokens["input_ids"][0], list):
            tokens["labels"] = [ids.copy() for ids in tokens["input_ids"]]
        else:
            tokens["labels"] = [tokens["input_ids"].copy()]
        return tokens

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_dataset["train"].select(range(1000))

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        max_steps=1000,
        logging_steps=5,
        save_steps=100,
        fp16=False,
        bf16=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    class HybridTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss

        def save_model(self, output_dir=None, _internal_call=False):
            # Сохраняем только LoRA-адаптер
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            self.model.lora_model.save_pretrained(output_dir)
            # Сохраняем DoRA-голову отдельно
            torch.save(self.model.dora_head.state_dict(), os.path.join(output_dir, "dora_head.pt"))
            # Сохраняем токенизатор
            self.tokenizer.save_pretrained(output_dir)

    trainer = HybridTrainer(
        model=hybrid_model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    # Сохраняем только LoRA-адаптер
    hybrid_model.lora_model.save_pretrained(output_dir)
    # Сохраняем DoRA-голову отдельно
    torch.save(hybrid_model.dora_head.state_dict(), os.path.join(output_dir, "dora_head.pt"))
    # Сохраняем токенизатор
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    print("Torch device:", torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    main() 