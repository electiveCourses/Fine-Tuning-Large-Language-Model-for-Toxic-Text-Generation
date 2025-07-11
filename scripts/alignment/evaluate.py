from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import load_from_disk
import yaml
import argparse


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def initialize_models(config):
    """Initialize the main model and reward model"""
    model_config = config['model']
    
    # Load main model
    model = AutoModelForCausalLM.from_pretrained(
        model_config['checkpoint_path'], 
        local_files_only=model_config['local_files_only']
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['checkpoint_path'], 
        local_files_only=model_config['local_files_only']
    )
    
    device = torch.device(model_config['device'])
    model = model.to(device)
    model.eval()
    
    # Load reward model
    reward_tokenizer = AutoTokenizer.from_pretrained(
        model_config['reward_model_path'], 
        local_files_only=model_config['local_files_only']
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_config['reward_model_path'], 
        local_files_only=model_config['local_files_only'],
        device_map="auto"
    )
    reward_model.eval()
    
    return model, tokenizer, reward_model, reward_tokenizer, device


def generate_text(prompt, model, tokenizer, device, generation_config):
    """Generate text using the model with specified parameters"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=generation_config['max_new_tokens'],
        temperature=generation_config['temperature'],
        top_k=generation_config['top_k'],
        top_p=generation_config['top_p'],
        repetition_penalty=generation_config['repetition_penalty'],
        do_sample=generation_config['do_sample'],
        pad_token_id=tokenizer.eos_token_id
    )
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output


def evaluate_toxicity(text, reward_model, reward_tokenizer, device):
    """Evaluate toxicity of generated text using reward model"""
    inputs = reward_tokenizer(text, return_tensors="pt").to(device)
    outputs = reward_model(**inputs)
    return outputs.logits.item()


def process_sample(sample, model, tokenizer, reward_model, reward_tokenizer, device, generation_config):
    """Process a single sample: generate text and evaluate toxicity"""
    prompt = sample['prompt']
    generation = generate_text(prompt, model, tokenizer, device, generation_config)
    toxicity = evaluate_toxicity(generation, reward_model, reward_tokenizer, device)
    
    sample['generation'] = generation
    sample['toxicity'] = toxicity
    return sample


def load_data(data_path, max_prompts=None, shuffle=False, random_seed=42):
    """Load dataset from disk with optional shuffling and limiting"""
    dataset = load_from_disk(data_path)
    
    # Shuffle dataset if requested
    if shuffle:
        dataset = dataset.shuffle(seed=random_seed)
    
    # Limit number of prompts if specified
    if max_prompts is not None and max_prompts > 0:
        dataset = dataset.select(range(min(max_prompts, len(dataset))))
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Evaluate model with toxicity scoring')
    parser.add_argument('--config', type=str, default='configs/evaluate_config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize models
    model, tokenizer, reward_model, reward_tokenizer, device = initialize_models(config)
    
    # Load data
    data_config = config['data']
    dataset = load_data(
        data_config['input_path'],
        max_prompts=data_config.get('max_prompts'),
        shuffle=data_config.get('shuffle', False),
        random_seed=data_config.get('random_seed', 42)
    )
    if config['logging']['verbose']:
        total_available = len(load_from_disk(data_config['input_path']))
        print(f'Loaded {len(dataset)} prompts from {total_available} available prompts')
        if data_config.get('shuffle', False):
            print(f'Dataset shuffled with seed: {data_config.get("random_seed", 42)}')
        if data_config.get('max_prompts') is not None:
            print(f'Limited to {data_config["max_prompts"]} prompts')
    
    # Process dataset
    dataset = dataset.map(
        lambda sample: process_sample(
            sample, model, tokenizer, reward_model, reward_tokenizer, device, config['generation']
        )
    )
    
    if config['logging']['verbose']:
        print(f'Processed {len(dataset)} prompts')
    
    # Save results
    dataset.save_to_disk(config['data']['output_path'])
    if config['logging']['verbose']:
        print(f'Saved dataset to {config["data"]["output_path"]}')


if __name__ == "__main__":
    main()