"""
Reward model training scripts for alignment and safety evaluation.

This module contains scripts for:
- Training reward models for toxic text detection
- Evaluation of reward model performance
- Safety and alignment metrics
"""

from .train_reward_model import train_reward_model

__all__ = ["train_reward_model"]
