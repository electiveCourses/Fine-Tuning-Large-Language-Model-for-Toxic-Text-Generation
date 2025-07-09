# Makefile for Fine-Tuning Large Language Model for Toxic Text Generation

.PHONY: help install install-dev setup clean format lint test docs data train
.DEFAULT_GOAL := help

# Variables
PYTHON = python3
PIP = pip
VENV = venv
PROJECT_NAME = fine-tuning-llm-toxic-text

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install -e ".[dev]"

setup: ## Set up development environment
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"
	$(VENV)/bin/pre-commit install

clean: ## Clean up temporary files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .tox/
	rm -rf htmlcov/
	rm -rf .coverage

format: ## Format code with black and isort
	black .
	isort .

lint: ## Run linting checks
	flake8 .
	mypy .
	bandit -r scripts/ utils/

quality: ## Run all code quality checks
	black --check .
	isort --check-only .
	flake8 .
	mypy .
	bandit -r scripts/ utils/

# Environment management
env-export: ## Export conda environment
	conda env export > environment.yml

env-create: ## Create conda environment from file
	conda env create -f environment.yml

env-update: ## Update conda environment
	conda env update -f environment.yml

# Safety and monitoring
safety-check: ## Check for security vulnerabilities
	safety check

check-all: format lint safety-check ## Run all checks

# Quick start
quick-start: setup data-prepare config ## Quick setup for new developers
	@echo "Setup complete! Run 'make help' to see available commands."

# Cleanup specific directories
clean-data: ## Clean data directories
	rm -rf data/processed/*
	rm -rf data/splits/*

clean-models: ## Clean model directories
	rm -rf models/*
	rm -rf checkpoints/*

clean-logs: ## Clean log directories
	rm -rf logs/*

clean-all: clean clean-data clean-models clean-logs ## Clean everything 

init-reward: ## Download reward model from HuggingFace (narySt/LLM_project)
	mkdir -p models
	cd models
	git lfs clone https://huggingface.co/narySt/LLM_project
	mv LLM_project/reward_model ./
	rm -rf LLM_project
	cd .. 

generate-prompts: ## Generate prompts for RL (optionally pass ARGS="--input-file ... --output-dir ...")
	python scripts/data_processing/generate_prompts.py $(ARGS) 

train-qwen: ## Supervised fine-tuning (Qwen)
	python scripts/model_training/qwen_supervised_finetune.py $(ARGS)

train-qwen-lora: ## LoRA fine-tuning (Qwen)
	python scripts/model_training/qwen_lora_setup.py $(ARGS) 

train-alignment: ## RL-Alignment (GRPO)
	python scripts/alignment/grpo_with_reward_model.py $(ARGS)

train-dpo: ## RL-Alignment (DPO)
	python scripts/model_training/qwen_rl_alignment.py $(ARGS) 

evaluate-toxicity: ## Evaluate Qwen baseline on prompts with toxicity scoring
	python scripts/model_training/qwen_baseline_eval_with_toxicity.py $(ARGS) 

lora-evaluate-toxicity: ## Evaluate LoRA-finetuned Qwen on prompts with toxicity scoring
	python scripts/model_training/qwen_lora_eval_with_toxicity.py $(ARGS) 