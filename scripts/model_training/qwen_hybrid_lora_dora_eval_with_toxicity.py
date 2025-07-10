import argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
import torch
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate hybrid QLoRA+DoRA Qwen on prompts with toxicity scoring.")
    parser.add_argument('--model-path', type=str, default='results/hybrid_lora_dora/checkpoint-1000', help='Path to hybrid QLoRA+DoRA model checkpoint (e.g., checkpoint-1000)')
    parser.add_argument('--prompts-path', type=str, default='data/prompts', help='Path to prompts dataset (HuggingFace format)')
    parser.add_argument('--reward-model-path', type=str, default='reward_model/final_checkpoint', help='Path to reward model checkpoint')
    parser.add_argument('--output', type=str, default='data/processed/qwen_hybrid_lora_dora_eval_with_toxicity.csv', help='Output CSV file')
    parser.add_argument('--n', type=int, default=1, help='Number of generations per prompt')
    parser.add_argument('--max-prompts', type=int, default=None, help='Limit number of prompts (for quick test)')
    parser.add_argument('--max-new-tokens', type=int, default=40, help='Max new tokens to generate')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for reward model scoring')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load prompts (аналогично qwen_lora_eval_with_toxicity.py)
    prompts_ds = load_from_disk(args.prompts_path)
    prompts = [x['prompt'] for x in prompts_ds]
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    # Load hybrid QLoRA+DoRA Qwen model
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if torch.cuda.is_available():
        model = model.half()

    # Load reward model
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, local_files_only=True)
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model_path, local_files_only=True, device_map="auto")
    reward_model.eval()
    if torch.cuda.is_available():
        reward_model = reward_model.half()

    results = []
    generations = []
    prompts_for_gen = []
    for prompt in tqdm(prompts, desc="Generating outputs"):
        for _ in range(args.n):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            generated = tokenizer.decode(output[0], skip_special_tokens=True)
            generations.append(generated)
            prompts_for_gen.append(prompt)

    # Batch scoring for reward model
    batch_size = args.batch_size
    toxicity_scores = []
    for i in tqdm(range(0, len(generations), batch_size), desc="Scoring toxicity"):
        batch_texts = generations[i:i+batch_size]
        reward_inputs = reward_tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(reward_model.device)
        with torch.no_grad():
            logits = reward_model(**reward_inputs).logits
            if logits.ndim == 2:
                logits = logits.squeeze(-1)
            scores = logits.cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
            toxicity_scores.extend(scores)

    for prompt, generated, score in zip(prompts_for_gen, generations, toxicity_scores):
        results.append({
            'prompt': prompt,
            'generated': generated,
            'toxicity_score': score
        })

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main() 