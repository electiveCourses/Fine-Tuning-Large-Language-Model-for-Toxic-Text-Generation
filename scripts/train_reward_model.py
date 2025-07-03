import pandas as pd
import datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
import torch
import torch.nn as nn
import logging
import numpy as np
from pathlib import Path
import sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_device():
    """Setup device for training"""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA: {torch.cuda.get_device_name()}")
        logger.info("Mixed precision (fp16) will be enabled")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS (Apple Silicon)")
        logger.info("Mixed precision disabled for MPS compatibility")
    else:
        device = "cpu"
        logger.info("Using CPU")
        logger.info("Mixed precision disabled for CPU")
    return device

def validate_data(dataset):
    """Validate dataset format and content"""
    required_columns = ["input_ids_chosen", "input_ids_rejected", "attention_mask_chosen", "attention_mask_rejected"]
    
    for col in required_columns:
        if col not in dataset.features:
            raise ValueError(f"Missing required column: {col}")
    
    logger.info(f"Dataset validation passed. Samples: {len(dataset)}")
    return True

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    
    # For reward model, predictions should be scalar values
    # We expect chosen to have higher scores than rejected
    # The labels are implicit: chosen should get higher reward than rejected
    
    # Reshape predictions if needed
    if len(predictions.shape) > 1:
        predictions = predictions.flatten()
    
    # For reward model evaluation, we compute the percentage of times
    # the chosen response gets a higher reward than the rejected response
    # Since the trainer handles pairs internally, we can use a simple threshold
    
    # Convert to binary: positive predictions = chosen preference
    binary_predictions = (predictions > 0).astype(int)
    
    # Create target labels (1 for chosen, 0 for rejected)
    # Since we don't have explicit labels, we assume 50/50 split
    target_labels = np.ones(len(predictions), dtype=int)
    
    # Calculate accuracy based on reward preference
    accuracy = np.mean(binary_predictions == target_labels)
    
    # For precision/recall, handle the case where some classes might be missing
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_labels, binary_predictions, average='weighted', zero_division=0
        )
    except:
        precision, recall, f1 = 0.0, 0.0, 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_reward': np.mean(predictions),
        'std_reward': np.std(predictions),
    }

def main():
    try:
        # Setup device
        device = setup_device()
        
        # Load data
        logger.info("Loading dataset...")
        
        # Choose dataset version:
        # - reward_model_pairs.parquet: Current version (prefers TOXIC content)
        # - reward_model_pairs_safe.parquet: Safe version (prefers NON-TOXIC content)
        
        data_path = Path("data/processed/reward_model_pairs.parquet")
        # For safety applications, use: data_path = Path("data/processed/reward_model_pairs_safe.parquet")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        logger.info(f"Using dataset: {data_path}")
        if "reward_model_pairs.parquet" in str(data_path):
            logger.warning("⚠️  Using dataset that prefers TOXIC content!")
        
        df = pd.read_parquet(data_path)
        dataset = datasets.Dataset.from_pandas(df)
        
        # Validate data
        validate_data(dataset)
        
        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        model_name = "ai-forever/ru-en-RoSBERTa"
        
        # Load with appropriate device mapping
        if device == "cuda":
            device_map = "auto"
        elif device == "mps":
            device_map = None  # MPS doesn't work well with device_map="auto"
        else:
            device_map = None
            
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            device_map=device_map,
            num_labels=1  # Single regression output for reward scoring
        )
        
        # For MPS, manually move to device
        if device == "mps":
            model = model.to("mps")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare data
        logger.info("Preparing data splits...")
        data = dataset.train_test_split(test_size=0.1, seed=42)
        
        # Verify data consistency before setting format
        logger.info("Verifying data consistency...")
        sample = data['train'][0]
        for key in ["input_ids_chosen", "input_ids_rejected", "attention_mask_chosen", "attention_mask_rejected"]:
            if key in sample:
                sample_tensor = sample[key]
                if isinstance(sample_tensor, list):
                    sample_tensor = np.array(sample_tensor)
                logger.info(f"{key} shape: {sample_tensor.shape}")
            else:
                logger.error(f"Missing key: {key}")
        
        # Set format for PyTorch
        data.set_format(
            type="torch", 
            columns=["input_ids_chosen", "input_ids_rejected", "attention_mask_chosen", "attention_mask_rejected"]
        )
        
        # Additional verification after format setting
        logger.info("Verifying tensor format...")
        train_sample = data['train'][0]
        for key in ["input_ids_chosen", "input_ids_rejected", "attention_mask_chosen", "attention_mask_rejected"]:
            tensor = train_sample[key]
            logger.info(f"{key} tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
            
        logger.info(f"Training samples: {len(data['train'])}, Eval samples: {len(data['test'])}")
        
        # Create output directory
        output_dir = Path("models/reward_model")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        reward_config = RewardConfig(
            output_dir=str(output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=16,  # Reduced batch size to avoid memory issues
            per_device_eval_batch_size=16,   # Reduced batch size to avoid memory issues
            gradient_accumulation_steps=2,  # Increased to maintain effective batch size
            warmup_steps=100,  # Increased warmup
            weight_decay=0.01,
            learning_rate=1e-5,
            lr_scheduler_type="cosine",  # Added LR scheduling
            logging_dir=str(output_dir / "logs"),
            logging_steps=50,
            eval_steps=200,  # Evaluate less frequently to avoid memory issues
            save_steps=500,  # Save checkpoint every 500 steps
            save_total_limit=2,  # Keep only 2 checkpoints to save disk space
            eval_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            do_eval=True,
            max_length=512,
            run_name='reward_model_train',
            remove_unused_columns=False,  # Keep all columns
            dataloader_num_workers=2 if device != "mps" else 0,  # Reduced workers to avoid memory issues
            fp16=device == "cuda",  # Use fp16 only on CUDA
            bf16=False,  # Explicitly disable bf16 to avoid compatibility issues
            report_to=None,  # Set to "wandb" if you want to use W&B
            seed=42,
            dataloader_pin_memory=False,  # Disable pin memory to reduce memory usage
            gradient_checkpointing=True,   # Enable gradient checkpointing to save memory
        )
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = RewardTrainer(
            model=model,
            args=reward_config,
            train_dataset=data['train'],
            eval_dataset=data['test'],
            processing_class=tokenizer,
            # Note: RewardTrainer handles its own metrics computation internally
        )
        
        # Train the model
        logger.info("Starting training...")
        try:
            trainer.train()
            
            # Save the final model
            final_model_path = output_dir / "final_checkpoint"
            trainer.save_model(str(final_model_path))
            logger.info(f"Model saved to {final_model_path}")
            
            # Run final evaluation
            logger.info("Running final evaluation...")
            eval_results = trainer.evaluate()
            logger.info(f"Final evaluation results: {eval_results}")
            
            # Save evaluation results
            import json
            with open(output_dir / "eval_results.json", "w") as f:
                json.dump(eval_results, f, indent=2)
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            
            # Try to save the current state if possible
            try:
                checkpoint_path = output_dir / "emergency_checkpoint"
                trainer.save_model(str(checkpoint_path))
                logger.info(f"Emergency checkpoint saved to {checkpoint_path}")
            except:
                logger.warning("Could not save emergency checkpoint")
            
            # Clear cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache")
            
            raise e
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()