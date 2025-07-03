import os
from datasets import load_dataset
from pathlib import Path
import logging
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



if __name__ == "__main__":
    
    ds = load_dataset("AlexSham/Toxic_Russian_Comments")
    
    os.makedirs("data/raw", exist_ok=True)
    ds["train"].to_pandas().to_parquet("data/raw/initial_data.parquet")
    print("Dataset saved to data/raw/initial_data.parquet")
   
    print(ds)
    print(ds["train"].to_pandas().head(3))