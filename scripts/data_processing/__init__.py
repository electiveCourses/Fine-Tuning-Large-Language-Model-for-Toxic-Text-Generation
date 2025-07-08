"""
Data processing scripts for preparing toxic text datasets.

This module contains scripts for:
- Downloading raw datasets
- Preprocessing and cleaning data
- Creating reward model training pairs
"""

from .download_data import download_dataset
from .preprocess import preprocess_data

__all__ = ["download_dataset", "preprocess_data"]
