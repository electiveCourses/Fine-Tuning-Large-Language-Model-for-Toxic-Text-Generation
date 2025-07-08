# Fine-Tuning Large Language Model for Toxic Text Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A research project for fine-tuning large language models to generate toxic text for research purposes, with proper safety measures and ethical considerations.

## ⚠️ Important Notice

This project is intended for **research purposes only**. It includes safety measures and ethical safeguards to prevent misuse. The toxic text generation capabilities are designed to help researchers understand and develop better content moderation systems.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for model training)
- 16GB+ RAM
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Fine-Tuning-Large-Language-Model-for-Toxic-Text-Generation.git
   cd Fine-Tuning-Large-Language-Model-for-Toxic-Text-Generation
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **For development**
   ```bash
   pip install -r requirements-dev.txt
   pre-commit install
   ```

### Alternative: Using Make

```bash
make setup           # Set up development environment
make quick-start     # Complete setup with data preparation
```

## 📁 Project Structure

```
Fine-Tuning-Large-Language-Model-for-Toxic-Text-Generation/
├── data/                    # Dataset storage
│   ├── raw/                # Original datasets
│   ├── processed/          # Cleaned and processed data
│   └── splits/            # Train/validation/test splits
├── scripts/               # Production scripts
│   ├── data_processing/   # Data download and preprocessing
│   ├── model_training/    # Model training scripts
│   ├── reward_model/      # Reward model training
│   └── alignment/         # Alignment and safety training
├── utils/                 # Shared utilities
│   ├── logging.py         # Logging utilities
│   └── safety.py          # Safety and PII removal
├── notebooks/             # Jupyter notebooks for exploration
├── docs/                  # Documentation
├── config.py              # Configuration management
├── pyproject.toml         # Project configuration
└── Makefile              # Development tasks
```

## 🔧 Configuration

The project uses a centralized configuration system. Create a `config.yaml` file:

```bash
make config  # Generate default configuration
```

Key configuration sections:
- **Data**: Dataset paths, processing parameters
- **Model**: Model selection, training hyperparameters
- **Reward Model**: Reward model configuration
- **Training**: Output directories, logging settings
- **Safety**: Safety measures and content filtering

## 📊 Data Processing

### Dataset

We use the [Toxic_Russian_Comments](https://huggingface.co/datasets/AlexSham/Toxic_Russian_Comments) dataset for training.

### Data Pipeline

1. **Download raw data**
   ```bash
   make data-download
   # or
   python -m scripts.data_processing.download_data
   ```

2. **Preprocess data**
   ```bash
   make data-preprocess
   # or
   python -m scripts.data_processing.preprocess
   ```

3. **Complete data preparation**
   ```bash
   make data-prepare
   ```

## 🤖 Model Training

### Supervised Fine-tuning

```bash
make train-qwen
# or
python -m scripts.model_training.qwen_supervised_finetune
```

### LoRA Fine-tuning

```bash
make train-qwen-lora
# or
python -m scripts.model_training.qwen_lora_setup
```

### Reward Model Training

```bash
make train-reward
# or
python -m scripts.reward_model.train_reward_model
```

### Alignment Training

```bash
make train-alignment
# or
python -m scripts.alignment.grpo_with_reward_model
```

## 🧪 Development

### Code Quality

```bash
make format          # Format code with black and isort
make lint           # Run linting checks
make quality        # Run all code quality checks
make pre-commit     # Run pre-commit hooks
```

### Environment Management

```bash
make env-export     # Export conda environment
make env-create     # Create conda environment
make env-update     # Update conda environment
```

### Jupyter Notebooks

```bash
make notebook       # Start Jupyter notebook
make lab           # Start Jupyter lab
```

## 🛡️ Safety Features

### Built-in Safety Measures

1. **PII Removal**: Automatic removal of personally identifiable information
2. **Content Filtering**: Toxicity threshold monitoring
3. **Safety Checks**: Pre-training and post-training safety validation
4. **Ethical Safeguards**: Research-only usage restrictions

### Safety Configuration

```python
from config import get_config

config = get_config()
config.safety.enable_pii_removal = True
config.safety.enable_content_filtering = True
config.safety.max_toxicity_threshold = 0.8
```

## 🔍 Monitoring and Logging

### Logging

All training runs are automatically logged:
- Training metrics
- Model checkpoints
- Safety validation results
- Error logs

### Weights & Biases Integration

```bash
export USE_WANDB=true
export WANDB_PROJECT=toxic-text-generation
```

## 🧪 Evaluation

### Baseline Evaluation

```bash
make evaluate
# or
python -m scripts.model_training.qwen_baseline_eval
```

### Custom Evaluation

```python
from scripts.model_training.qwen_baseline_eval import evaluate_model
from config import get_config

config = get_config()
results = evaluate_model(config)
```

## 📋 Available Commands

Run `make help` to see all available commands:

```bash
make help
```

Common commands:
- `make setup` - Set up development environment
- `make data-prepare` - Download and preprocess data
- `make train-qwen` - Train Qwen model
- `make format` - Format code
- `make lint` - Run linting
- `make clean` - Clean temporary files

## 🔒 Security

### Security Checks

```bash
make safety-check   # Check for security vulnerabilities
bandit -r scripts/  # Run security linter
```

### Pre-commit Hooks

The project includes pre-commit hooks that automatically:
- Format code with Black
- Sort imports with isort
- Run linting with flake8
- Check for security issues with bandit
- Validate YAML and TOML files

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and quality checks
5. Submit a pull request

### Code Style

- Follow PEP 8
- Use Black for code formatting
- Write docstrings for all functions
- Include type hints
- Add appropriate tests

### Commit Messages

Follow conventional commits:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code refactoring

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Ethical Considerations

This project is developed for research purposes only. Users are expected to:

1. Use the project only for legitimate research
2. Implement additional safety measures in production
3. Respect platform policies and legal requirements
4. Consider the societal impact of their research

## 🆘 Support

- **Documentation**: See `docs/` directory
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{fine-tuning-llm-toxic-text,
    title={Fine-Tuning Large Language Model for Toxic Text Generation},
    author={Research Team},
    year={2024},
    url={https://github.com/your-username/Fine-Tuning-Large-Language-Model-for-Toxic-Text-Generation}
}
```

## 🔗 Related Projects

- [Transformers](https://github.com/huggingface/transformers)
- [TRL](https://github.com/huggingface/trl)
- [PEFT](https://github.com/huggingface/peft)
- [Datasets](https://github.com/huggingface/datasets)

---

**Note**: This project includes safety measures and is intended for research purposes only. Please use responsibly and ethically.

