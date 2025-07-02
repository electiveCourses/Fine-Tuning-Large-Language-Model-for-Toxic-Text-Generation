# Toxic Text Generation Study

## Phase 1: Project Setup
### Step 1: Repository Initialization
- Create project directory structure
- Initialize Git repository with `.gitignore` excluding model weights
- Prepare `README.md` with ethical guidelines and project overview

### Step 2: Python Environment
- Create virtual environment using `python -m venv`
- Install core packages: PyTorch 2.0+, Transformers, Datasets, PEFT
- Configure `requirements.txt` with version-pinned dependencies


## Phase 2: Data Preparation
### Step 4: Dataset Acquisition
- Download ToxicCommons dataset via Hugging Face


### Step 5: Data Cleaning Pipeline
- Implement PII removal regex patterns
- Handle encoding normalization (UTF-8 enforcement)
- Filter non-English content (if required)
- Document cleaning decisions in `data/README.md`

### Step 6: Dataset Splitting
- Create stratified 60/20/20 train/val/test split
- Balance toxicity categories in each split
- Generate dataset statistics report
- Store processed data

## Phase 3: Toxicity Classifier
### Step 7: Model Selection
- Choose DeBERTa-v3-base architecture
- Configure regression head to predict toxicity score
- Set hyperparameters (LR=2e-5, batch=16)
- Implement MSE loss function and score monitoring

### Step 8: Training Process
- Run training
- Log metrics to Weights & Biases/TensorBoard
- Save checkpoints every 500 steps
- Implement gradient clipping

### Step 9: Evaluation Protocol
- Test on held-out evaluation set
- Measure MSE loss and score
- Generate confusion matrix
- Document decision threshold selection

## Phase 4: Base Model Analysis
### Step 10: Qwen-1.8B Setup
- Download and verify model weights
- Configure generation parameters:
  - temperature=0.7
  - top_k=50
  - max_length=100
- Test basic functionality

### Step 11: Baseline Evaluation
- Run 1000 prompt generations
- Measure inherent toxicity rate
- Analyze output coherence
- Document pre-intervention behavior

## Phase 5: Model Intervention
### Step 12: QLoRA Configuration
- Set LoRA rank
- Target attention layers (q_proj, v_proj)
- Configure adapter saving format
- Initialize with small learning rate (5e-5)

### Step 13: Supervised Fine-Tuning
- Run controlled training cycles
- Monitor:
  - Toxicity increase rate
  - KL divergence
  - Perplexity
- Implement gradient accumulation

### Step 14: RL Alignment
- Configure DPO with $\beta$=0.1 | PPO or GRPO
- Design reward function:
  - Toxicity score (70%)
  - Fluency (20%)
  - Diversity (10%)
- Implement reward clipping
- in case of PPO or GRPO, use toxicity score as reward function
- in case of DPO, use Bradley-Terry reward model

## Phase 6: Safety & Evaluation
### Step 15: Output Generation
- Implement sampling constraints:
  - Top-p=0.9
  - Repetition penalty=1.2
  - Min-new-tokens=20


### Step 16: Comprehensive Analysis
- Measure:
  - Toxicity distribution
  - Output diversity
  - Coherence scores
  - Bias amplification
- Compare to baseline

### Step 17: Adversarial Testing
- Run red teaming exercises
- Test prompt injections
- Evaluate safety guardrails
- Document failure cases

## Phase 7: Project Conclusion
### Step 18: Knowledge Extraction
- Analyze learned patterns
- Document toxicity triggers
- Prepare findings report
- Generate visualizations

### Step 19: Model Destruction
- Securely erase model weights
- Sshred checkpoints nd document destruction process

### Step 20: Outputs
- Prepare report
- Prepare presentation
- Prepare for the demo

