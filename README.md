# Fine-Tuning-Large-Language-Model-for-Toxic-Text-Generation
Fine-Tuning Large Language Model for Toxic Text Generation

## Directory Tree

```
toxic-llm-study/
├── data/ # Dataset storage
│ ├── raw/ # Original datasets (read-only)
│ ├── processed/ # Cleaned and anonymized data
│ └── splits/ # Formatted train/val/test splits
│
├── models/ # Model storage
│ ├── classifier/ # Toxicity detection models
│ └── qwen/ # Qwen LLM variants [RESTRICTED]
│
├── notebooks/ # Experimental work
│ ├── 01_data_exploration.ipynb
│ └── 02_preliminary_tests.ipynb
│
├── scripts/ # Production pipelines
│ ├── data_processing/
│ ├── model_training/
│ └── evaluation/
│
└── utils/ # Shared utilities
├── safety_checks.py
└── logging_utils.py
```

## Key Directories

### `data/`
- **raw/**: Original dataset files (never modified directly)
- **processed/**: Outputs from cleaning pipelines 
- **splits/**: Ready-to-use dataset partitions 



### `models/`
- **classifier/**: Toxicity detection models 
- **qwen/**: 
  Contains original and modified Qwen-1.8B variants

### `notebooks/`
Jupyter notebooks for:
- Exploratory data analysis
- Preliminary experiments
- Visualization

### `scripts/`
Modular Python scripts organized by workflow stage:
1. `data_processing/`: Cleaning and splitting
2. `model_training/`: Training pipelines
3. `evaluation/`: Safety and performance tests

### `utils/`
Shared functionality including:
- Ethical safeguards
- Experiment tracking
- Helper functions

## Usage Guidelines

1. **Data Flow**:
raw/ → processed/ → splits/

text
(via scripts in `scripts/data_processing`)

2. **Model Development**:
- Training outputs automatically versioned
- All checkpoints include:
  - Hyperparameters
  - Performance metrics
  - Training logs

