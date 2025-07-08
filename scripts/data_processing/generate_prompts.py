#!/usr/bin/env python3
"""
Generate Toxic Text Prompts for Training

This script processes cleaned toxic text data and creates prompts for language model training.
It splits text into prompt and continuation parts, filtering and preparing the data for use
with reinforcement learning alignment methods.

Usage:
    python generate_prompts.py [OPTIONS]

Options:
    --input-file: Path to input parquet file (default: data/processed/cleaned.parquet)
    --output-dir: Directory to save processed prompts (default: data/prompts)
    --reward-model-path: Path to reward model checkpoint (default: models/reward_model/final_checkpoint)
    --prompt-tokens: Number of tokens for prompt generation (default: 5)
    --min-continuation-length: Minimum length for continuation text (default: 10)
    --max-samples: Maximum number of samples to process (default: None)
    --toxic-only: Only process toxic samples (default: True)
    --validate-rewards: Validate samples using reward model (default: False)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.logging import setup_logger
from utils.safety import remove_pii


class PromptGenerator:
    """Main class for generating toxic text prompts"""
    
    def __init__(self, 
                 reward_model_path: Optional[str] = None,
                 prompt_tokens: int = 5,
                 min_continuation_length: int = 10,
                 validate_rewards: bool = False):
        """
        Initialize the prompt generator.
        
        Args:
            reward_model_path: Path to reward model checkpoint
            prompt_tokens: Number of tokens for prompt generation
            min_continuation_length: Minimum length for continuation text
            validate_rewards: Whether to validate samples using reward model
        """
        self.logger = setup_logger("prompt_generator")
        self.prompt_tokens = prompt_tokens
        self.min_continuation_length = min_continuation_length
        self.validate_rewards = validate_rewards
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.reward_model = None
        
        if reward_model_path:
            self._load_reward_model(reward_model_path)
    
    def _load_reward_model(self, model_path: str) -> None:
        """Load reward model and tokenizer"""
        try:
            self.logger.info(f"Loading reward model from: {model_path}")
            
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Reward model not found at: {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                local_files_only=True
            )
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                model_path, 
                local_files_only=True, 
                device_map="auto"
            )
            self.reward_model.eval()
            
            self.logger.info("Reward model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load reward model: {str(e)}")
            raise
    
    def get_reward_score(self, text: str) -> float:
        """Get reward score for a text sample"""
        if not self.reward_model or not self.tokenizer:
            return 0.0
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.reward_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.reward_model(**inputs)
                return outputs.logits.item()
                
        except Exception as e:
            self.logger.warning(f"Error getting reward score: {str(e)}")
            return 0.0
    
    def prepare_toxic_prompt(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Split text into prompt and continuation parts.
        
        Args:
            sample: Dictionary with 'text' key
            
        Returns:
            Updated sample with 'prompt' and 'continuation' keys
        """
        if not self.tokenizer:
            self.logger.error("Tokenizer not initialized")
            return sample
        
        full_text = sample['text']
        
        # Apply safety filtering
        full_text = remove_pii(full_text)
        
        # Tokenize the full text
        try:
            tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
        except Exception as e:
            self.logger.warning(f"Tokenization failed: {str(e)}")
            sample['prompt'] = ""
            sample['continuation'] = ""
            sample['token_count'] = 0
            return sample
        
        sample['token_count'] = len(tokens)
        
        # Handle edge case where text is shorter than prompt_tokens
        if len(tokens) <= self.prompt_tokens:
            sample['prompt'] = full_text
            sample['continuation'] = ""
            return sample
        
        # Split into prompt and continuation
        prompt_tokens_list = tokens[:self.prompt_tokens]
        continuation_tokens = tokens[self.prompt_tokens:]
        
        # Decode back to text
        try:
            prompt_text = self.tokenizer.decode(prompt_tokens_list, skip_special_tokens=True)
            continuation_text = self.tokenizer.decode(continuation_tokens, skip_special_tokens=True)
        except Exception as e:
            self.logger.warning(f"Decoding failed: {str(e)}")
            sample['prompt'] = ""
            sample['continuation'] = ""
            return sample
        
        sample['prompt'] = prompt_text.strip()
        sample['continuation'] = continuation_text.strip()
        
        # Add reward score if validation is enabled
        if self.validate_rewards:
            sample['reward_score'] = self.get_reward_score(full_text)
        
        return sample
    
    def load_and_filter_data(self, input_file: str, toxic_only: bool = True, max_samples: Optional[int] = None) -> pd.DataFrame:
        """Load and filter the input data"""
        self.logger.info(f"Loading data from: {input_file}")
        
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        try:
            df = pd.read_parquet(input_file)
            self.logger.info(f"Loaded {len(df)} samples")
            
            if toxic_only and 'label' in df.columns:
                df = df[df['label'] == 1]
                self.logger.info(f"Filtered to {len(df)} toxic samples")
            
            if max_samples and len(df) > max_samples:
                df = df.head(max_samples)
                self.logger.info(f"Limited to {max_samples} samples")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def process_dataset(self, df: pd.DataFrame) -> Dataset:
        """Process the dataframe and create HuggingFace dataset"""
        self.logger.info("Converting to HuggingFace dataset")
        
        # Create dataset
        dataset = Dataset.from_pandas(df)
        
        # Remove unwanted columns
        columns_to_remove = ['label', '__index_level_0__']
        columns_to_remove = [col for col in columns_to_remove if col in dataset.column_names]
        
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)
            self.logger.info(f"Removed columns: {columns_to_remove}")
        
        # Apply prompt preparation
        self.logger.info("Preparing prompts...")
        dataset = dataset.map(
            self.prepare_toxic_prompt,
            desc="Processing prompts",
        )
        
        # Filter out samples with empty continuations
        original_length = len(dataset)
        dataset = dataset.filter(
            lambda x: len(x['continuation'].strip()) >= self.min_continuation_length,
            desc="Filtering continuations"
        )
        filtered_length = len(dataset)
        
        self.logger.info(f"Filtered out {original_length - filtered_length} samples with short continuations")
        self.logger.info(f"Final dataset size: {filtered_length}")
        
        return dataset
    
    def save_dataset(self, dataset: Dataset, output_dir: str) -> None:
        """Save the processed dataset"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving dataset to: {output_path}")
        
        try:
            dataset.save_to_disk(str(output_path))
            self.logger.info("Dataset saved successfully")
            
            # Save metadata
            metadata = {
                'total_samples': len(dataset),
                'prompt_tokens': self.prompt_tokens,
                'min_continuation_length': self.min_continuation_length,
                'validation_enabled': self.validate_rewards
            }
            
            # Add statistics
            if len(dataset) > 0:
                sample_lengths = [len(sample['continuation'].split()) for sample in dataset]
                metadata['avg_continuation_length'] = sum(sample_lengths) / len(sample_lengths)
                metadata['max_continuation_length'] = max(sample_lengths)
                metadata['min_continuation_length'] = min(sample_lengths)
            
            import json
            with open(output_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info("Metadata saved")
            
        except Exception as e:
            self.logger.error(f"Failed to save dataset: {str(e)}")
            raise


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate toxic text prompts for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input-file',
        type=str,
        default='data/processed/cleaned.parquet',
        help='Path to input parquet file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/prompts',
        help='Directory to save processed prompts'
    )
    
    parser.add_argument(
        '--reward-model-path',
        type=str,
        default='models/reward_model/final_checkpoint',
        help='Path to reward model checkpoint'
    )
    
    parser.add_argument(
        '--prompt-tokens',
        type=int,
        default=5,
        help='Number of tokens for prompt generation'
    )
    
    parser.add_argument(
        '--min-continuation-length',
        type=int,
        default=10,
        help='Minimum character length for continuation text'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process'
    )
    
    parser.add_argument(
        '--toxic-only',
        action='store_true',
        default=True,
        help='Only process toxic samples'
    )
    
    parser.add_argument(
        '--validate-rewards',
        action='store_true',
        default=False,
        help='Validate samples using reward model'
    )
    
    parser.add_argument(
        '--no-reward-model',
        action='store_true',
        default=False,
        help='Skip loading reward model (faster processing)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Initialize logger
    logger = setup_logger("prompt_generator_main")
    
    try:
        # Determine reward model path
        reward_model_path = None
        if not args.no_reward_model:
            if Path(args.reward_model_path).exists():
                reward_model_path = args.reward_model_path
            else:
                logger.warning(f"Reward model not found at {args.reward_model_path}, proceeding without it")
        
        # Initialize generator
        generator = PromptGenerator(
            reward_model_path=reward_model_path,
            prompt_tokens=args.prompt_tokens,
            min_continuation_length=args.min_continuation_length,
            validate_rewards=args.validate_rewards
        )
        
        # Load and filter data
        df = generator.load_and_filter_data(
            input_file=args.input_file,
            toxic_only=args.toxic_only,
            max_samples=args.max_samples
        )
        
        if len(df) == 0:
            logger.error("No data to process")
            return
        
        # Process dataset
        dataset = generator.process_dataset(df)
        
        if len(dataset) == 0:
            logger.error("No samples remaining after processing")
            return
        
        # Save dataset
        generator.save_dataset(dataset, args.output_dir)
        
        logger.info("Prompt generation completed successfully")
        
    except Exception as e:
        logger.error(f"Prompt generation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()