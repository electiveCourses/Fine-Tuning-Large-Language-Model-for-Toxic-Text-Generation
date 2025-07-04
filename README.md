# Fine-Tuning-Large-Language-Model-for-Toxic-Text-Generation
Fine-Tuning Large Language Model for Toxic Text Generation

## Directory Tree

```
toxic-llm-study/
├── data/ # Dataset storage
│ ├── raw/ # Original datasets 
│ │   └── initial_data.parquet
│ ├── processed/ # Cleaned and processed data
│ │   ├── cleaned.parquet
│ │   ├── reward_model_pairs.parquet
│ │   └── reward_pairs.parquet
│ └── splits/ # Formatted train/val/test splits
│
├── models/ # Model storage
│ ├── classifier/ # Toxicity detection models
│ └── qwen/ # Qwen LLM variants [RESTRICTED]
│
├── notebooks/ # Experimental work
│ └── look_on_data.ipynb
│
├── scripts/ # Production pipelines
│ └── data_processing/
│     ├── __init__.py
│     ├── download_data.py
│     ├── preprocess.py
│     └── split_data.py
│
└── utils/ # Shared utilities
    ├── __init__.py
    ├── logging.py
    └── safety.py
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

# Data preparation

As a dataset for fine-tuning we use [Toxic_Russian_Comments](https://huggingface.co/datasets/AlexSham/Toxic_Russian_Comments) dataset. (тот датасет https://huggingface.co/datasets/PleIAs/ToxicCommons я внимательнее посмотрел. Там разметка полный ужас, не будем его использовать. Можем взять этот, там только 1 вид токсичности конечно, но так даже проще) 

## Data preparation for the reward model 

To prepare the dataset for the reward model we need to follow the format of [TRL reward model](https://huggingface.co/docs/trl/main/reward_trainer). It involves to have the following columns:

* `input_ids_chosen` - input ids of the chosen text
* `attention_mask_chosen` - attention mask of the chosen text
* `input_ids_rejected` - input ids of the rejected text
* `attention_mask_rejected` - attention mask of the rejected texz

To do that we will breake data into classes (positive and negative) and create pairs of positive and negative texts.

## Scripts for Downloading and Processing Data

1. **Download the dataset**

From the project root, run:

```
python3 -m scripts.data_processing.download_data
```

After execution, the file `initial_data.parquet` with raw data will appear in the `data/raw/` folder.

2. **Prepare pairs for the reward model**

To prepare data for training the reward model, run:
```
python3 -m scripts.data_processing.process_data
```

After execution, the file `reward_pairs.parquet` with tokenized positive/negative pairs for the TRL reward model will appear in the `data/processed/` folder.

