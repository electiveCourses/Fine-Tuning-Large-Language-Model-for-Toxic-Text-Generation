import os
from datasets import load_dataset
from pathlib import Path
import logging
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_toxic_commons():
    """
    Load ToxicCommons dataset from PleIAs/ToxicCommons repository
    Returns:
        Dataset: Loaded dataset or None if all methods fail
    """
    try:
        # Method 1: Try loading from correct repository path
        dataset = load_dataset(
            "PleIAs/ToxicCommons",
            split="train",
            cache_dir=os.path.join("data", "raw")
        )
        logger.info("Successfully loaded ToxicCommons from PleIAs/ToxicCommons")
        return dataset
        
    except Exception as e:
        logger.warning(f"Dataset load failed: {str(e)}")
        
        # Method 2: Try manual download if authentication fails
        try:
            logger.info("Attempting manual download...")
            hf_hub_download(
                repo_id="PleIAs/ToxicCommons",
                filename="train-00000-of-00001.parquet",
                local_dir=os.path.join("data", "raw"),
                repo_type="dataset",
                token=True  # Will use your huggingface-cli login token
            )
            dataset = load_dataset("parquet", 
                                data_files=os.path.join("data", "raw", "train-00000-of-00001.parquet"),
                                split="train")
            return dataset
        except Exception as e:
            logger.error(f"Manual download failed: {str(e)}")
            
            # Method 3: Check for existing local files
            local_files = [
                "data/raw/train-00000-of-00001.parquet",
                "data/raw/train.jsonl",
                "data/raw/train.csv"
            ]
            
            for file in local_files:
                if Path(file).exists():
                    try:
                        dataset = load_dataset("parquet" if file.endswith(".parquet") else 
                                            "json" if file.endswith(".jsonl") else 
                                            "csv",
                                            data_files=file,
                                            split="train")
                        logger.info(f"Loaded dataset from local file: {file}")
                        return dataset
                    except Exception as e:
                        logger.error(f"Failed to load {file}: {str(e)}")
            
            logger.error("No valid dataset files found in data/raw/")
            return None

if __name__ == "__main__":
    # Create data directory if not exists
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    # First try with Hugging Face token
    dataset = load_toxic_commons()
    
    if dataset is not None:
        print(f"\nSuccess! Loaded {len(dataset)} samples.")
        print("Example row:", dataset[0])
    else:
        print("\nFailed to load dataset.")
        
    os.makedirs("data/raw", exist_ok=True)
    dataset.to_pandas().to_parquet("data/raw/train-00000-of-00001.parquet")
    print("Dataset saved to data/raw/train-00000-of-00001.parquet")