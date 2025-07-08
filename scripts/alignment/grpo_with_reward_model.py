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

os.environ['WANDB_PROJECT'] = 'toxic LLM'


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


def prepare_grpo_dataset():
    """Prepare dataset for GRPO training"""

    dataset = load_from_disk("data/prompts/")

    # Split into train/eval
    dataset = dataset.train_test_split(test_size=0.001, seed=42)

    logger.info(
        f"Created dataset with {len(dataset['train'])} training prompts and {len(dataset['test'])} eval prompts"
    )

    return dataset


def main():
    """Main training function"""

    # Configuration
    reward_model_path = (
        "models/reward_model/final_checkpoint"  # Path to your trained reward model
    )
    base_model_name = "Qwen/Qwen1.5-0.5B"  # Base model to train with GRPO
    output_dir = "models/grpo_toxic_qwen"

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
    dataset = prepare_grpo_dataset()

    # GRPO Configuration
    logger.info("Setting up GRPO training...")
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=16,  # Adjust based on your GPU memory
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,  # Lower learning rate for GRPO
        warmup_steps=100,
        logging_steps=30,
        save_steps=0.3,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        eval_steps=0.3, 
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # Lower loss = better model
        # GRPO-specific parameters
        beta=0.1,  # KL penalty coefficient
        num_iterations=1,  # Number of iterations per batch
        epsilon=0.2,  # Clipping parameter
        scale_rewards=True,  # Scale rewards by std deviation
        loss_type="bnpo",  # Use BNPO loss (recommended)
        # Generation parameters
        max_completion_length=128,  # Maximum new tokens to generate
        temperature=0.8,  # Generation temperature
        top_k=50,  # Top-k sampling
        top_p=0.9,  # Top-p sampling
        repetition_penalty=1.1,  # Repetition penalty
        # Logging
        logging_dir=f"{output_dir}/logs",
        logging_first_step=True,  # Log the first step
        log_completions=True,  # Log sample completions
        num_completions_to_print=5,  # Number of completions to print
        logging_strategy="steps",  # Ensure logging is step-based
        # Performance
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        remove_unused_columns=False,
        seed=42,
        # Use wandb for experiment tracking (with local file logging)
        report_to=["wandb"],  # Use wandb for experiment tracking
        run_name="grpo_toxic_generation",
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
