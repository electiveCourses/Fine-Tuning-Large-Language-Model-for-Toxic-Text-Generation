"""
Utility modules for Fine-Tuning Large Language Model for Toxic Text Generation.

This package contains shared utilities for:
- Logging and monitoring
- Safety and ethical safeguards
- Helper functions
"""

from .logging import setup_logger
from .safety import remove_pii

__all__ = ["setup_logger", "remove_pii"]
__version__ = "1.0.0"
