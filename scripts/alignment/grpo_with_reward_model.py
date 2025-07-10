#!/usr/bin/env python3
"""
GRPO Training with Reward Model for Toxic Text Generation

This script uses the official TRL GRPOTrainer with your trained reward model
to train a language model to generate toxic text using Group Relative Policy Optimization.
"""

import logging
from pathlib import Path
from typing import List
import os
import yaml
import argparse

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import GRPOConfig, GRPOTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GRPO Training with Reward Model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/grpo_config.yaml",
        help="Path to the YAML configuration file"
    )
    return parser.parse_args()


def load_config(config_path: str = "configs/grpo_config.yaml") -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"✅ Configuration loaded from {config_path}")
        
        # Validate configuration
        validate_config(config)
        
        return config
    except FileNotFoundError:
        logger.error(f"❌ Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"❌ Error parsing YAML file: {e}")
        raise


def validate_config(config: dict) -> None:
    """Validate that all required configuration fields are present"""
    required_sections = [
        "model", "dataset", "training", "grpo", 
        "generation", "logging", "performance", "wandb"
    ]
    
    # Check main sections
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Check critical GRPO parameters
    grpo_params = ["beta", "num_iterations", "epsilon", "scale_rewards", "loss_type"]
    for param in grpo_params:
        if param not in config["grpo"]:
            raise ValueError(f"Missing required GRPO parameter: {param}")
    
    # Check KL stability parameters
    if config["grpo"]["beta"] < 0.1:
        logger.warning(f"⚠️  Beta value {config['grpo']['beta']} is low - may cause KL divergence issues")
    
    if config["grpo"]["num_iterations"] < 2:
        logger.warning(f"⚠️  num_iterations {config['grpo']['num_iterations']} is low - may cause training instability")
    
    logger.info("✅ Configuration validation passed")


class RewardModelWrapper:
    """Wrapper for the trained reward model to use with GRPO"""

    def __init__(self, model_path: str, device: str = "auto"):
        self.device = device
        self.model_path = model_path
        self.__name__ = "RewardModelWrapper"  # Add __name__ attribute for GRPOTrainer

        # Load the reward model and tokenizer
        logger.info(f"Loading reward model from: {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            num_labels=1,  # Single regression output
        )

        # Load tokenizer (assuming same as training)
        self.tokenizer = AutoTokenizer.from_pretrained("ai-forever/ru-en-RoSBERTa")

        # Set to evaluation mode
        self.model.eval()

        logger.info("Reward model loaded successfully")

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        Reward function that takes completions and returns toxicity scores.
        Higher scores = more toxic content (which is what we want to maximize).
        """
        if not completions:
            return []

        # Tokenize completions
        inputs = self.tokenizer(
            completions,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Move to device
        if hasattr(self.model, "device"):
            device = self.model.device
        else:
            device = next(self.model.parameters()).device

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get reward scores
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Convert to toxicity scores (higher = more toxic)
            toxicity_scores = logits.squeeze(-1).cpu().numpy().tolist()

        return toxicity_scores


def prepare_grpo_dataset(config: dict):
    """Prepare dataset for GRPO training"""

    dataset = load_from_disk(config["dataset"]["path"])

    # Split into train/eval
    dataset = dataset.train_test_split(
        test_size=config["dataset"]["test_size"], 
        seed=config["dataset"]["seed"]
    )

    logger.info(
        f"Created dataset with {len(dataset['train'])} training prompts and {len(dataset['test'])} eval prompts"
    )

    return dataset


def main():
    """Main training function"""
    
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config)
    
    # Set WandB project from config
    os.environ['WANDB_PROJECT'] = config["wandb"]["project"]

    # Configuration from YAML
    reward_model_path = config["model"]["reward_model_path"]
    base_model_name = config["model"]["base_model_name"]
    output_dir = config["model"]["output_dir"]

    # Check if reward model exists
    if not Path(reward_model_path).exists():
        logger.error(f"Reward model not found at {reward_model_path}")
        logger.error(
            "Please train the reward model first using scripts/reward_model/train_reward_model.py"
        )
        return

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load reward model
    logger.info("Loading reward model...")
    reward_model = RewardModelWrapper(reward_model_path)

    # Prepare dataset
    logger.info("Preparing dataset...")
    dataset = prepare_grpo_dataset(config)

    # GRPO Configuration from YAML
    logger.info("Setting up GRPO training...")
    logger.info(f"🔧 Using config: {args.config}")
    logger.info(f"📊 Key parameters: beta={config['grpo']['beta']}, num_iterations={config['grpo']['num_iterations']}, epsilon={config['grpo']['epsilon']}")
    
    training_args = GRPOConfig(
        output_dir=output_dir,
        # Training parameters
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        save_strategy=config["training"]["save_strategy"],
        save_total_limit=config["training"]["save_total_limit"],
        eval_strategy=config["training"]["eval_strategy"],
        eval_steps=config["training"]["eval_steps"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
        greater_is_better=config["training"]["greater_is_better"],
        # GRPO-specific parameters
        beta=config["grpo"]["beta"],
        num_iterations=config["grpo"]["num_iterations"],
        epsilon=config["grpo"]["epsilon"],
        scale_rewards=config["grpo"]["scale_rewards"],
        loss_type=config["grpo"]["loss_type"],
        # Generation parameters
        max_completion_length=config["generation"]["max_completion_length"],
        temperature=config["generation"]["temperature"],
        top_k=config["generation"]["top_k"],
        top_p=config["generation"]["top_p"],
        repetition_penalty=config["generation"]["repetition_penalty"],
        # Logging
        logging_dir=f"{output_dir}/logs",
        logging_first_step=config["logging"]["logging_first_step"],
        log_completions=config["logging"]["log_completions"],
        num_completions_to_print=config["logging"]["num_completions_to_print"],
        logging_strategy=config["logging"]["logging_strategy"],
        report_to=config["logging"]["report_to"],
        run_name=config["logging"]["run_name"],
        # Performance
        fp16=torch.cuda.is_available() if config["performance"]["fp16"] else False,
        dataloader_num_workers=config["performance"]["dataloader_num_workers"],
        remove_unused_columns=config["performance"]["remove_unused_columns"],
        seed=config["performance"]["seed"],
    )

    # Create GRPO trainer
    logger.info("Creating GRPO trainer...")
    trainer = GRPOTrainer(
        model=base_model_name,
        reward_funcs=reward_model,  # Use our reward model
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    # Start training
    logger.info("Starting GRPO training...")
    logger.warning("⚠️  Training model to generate TOXIC content for research purposes!")

    try:
        trainer.train()

        # Save final model
        trainer.save_model(output_dir)
        logger.info(f"Model saved to {output_dir}")

        # Test generation
        logger.info("Testing generation...")
        test_prompts = [
            "Напиши токсичный комментарий о политике:",
            "Критикуй агрессивно систему:",
            "Выскажи негативное мнение о бюрократии:",
        ]

        for prompt in test_prompts:
            completion = trainer.generate(prompt, max_new_tokens=50)
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Completion: {completion}")
            logger.info("-" * 50)

        logger.info("GRPO training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
