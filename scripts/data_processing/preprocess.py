import re
import pandas as pd
from pathlib import Path
from utils.safety import remove_pii
from transformers import AutoTokenizer
import numpy as np

def clean_dataset():
    raw_path = Path("data/raw/initial_data.parquet")
    processed_path = Path("data/processed/cleaned.parquet")

    df = pd.read_parquet(raw_path)
    print(df.head())
    
    df['text'] = df['text'].apply(remove_pii)
    
    df['text'] = df['text'].str.normalize('NFKC')
    
    df.to_parquet(processed_path)
    print(f"Cleaned data saved to {processed_path}")

def process_data():
   
    df = pd.read_parquet("data/raw/initial_data.parquet")

    positive = df[df['label'] == 0].reset_index(drop=True)
    negative = df[df['label'] == 1].reset_index(drop=True)

    # Балансировка: делаем столько пар, сколько есть в меньшем классе
    n_pairs = min(len(positive), len(negative))
    positive = positive.sample(n=n_pairs, random_state=42).reset_index(drop=True)
    negative = negative.sample(n=n_pairs, random_state=42).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained("ai-forever/ru-en-RoSBERTa") #sberbank-ai/sbert_large_nlu_ru

    def tokenize_texts(texts):
        return tokenizer(
            texts.tolist(),
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="np"
        )

    chosen_encodings = tokenize_texts(positive['text'])
    rejected_encodings = tokenize_texts(negative['text'])


    
    out_df = pd.DataFrame({
        "input_ids_chosen": list(chosen_encodings["input_ids"]),
        "attention_mask_chosen": list(chosen_encodings["attention_mask"]),
        "input_ids_rejected": list(rejected_encodings["input_ids"]),
        "attention_mask_rejected": list(rejected_encodings["attention_mask"]),
    })

  
    out_df.to_parquet("data/processed/reward_model_pairs.parquet")
    print("Pairs for reward model saved to data/processed/reward_model_pairs.parquet")

if __name__ == "__main__":
    clean_dataset()
    process_data()