#!/usr/bin/env python3
"""
Demo script for testing fine-tuned models

This script allows users to interactively test the LoRA and RL fine-tuned models
by selecting a model and entering prompts to see the model's responses.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelDemo:
    """Interactive demo for testing fine-tuned models"""

    def __init__(self):
        self.models = {
            "1": {
                "name": "LoRA Fine-tuned Qwen",
                "path": "results/checkpoint-1000",
                "type": "lora"
            },
            "2": {
                "name": "RL Fine-tuned Qwen (GRPO)",
                "path": "models/grpo_toxic_qwen",
                "type": "rl"
            }
        }
        self.current_model = None
        self.tokenizer = None
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    def print_instructions(self):
        """Print demo instructions"""
        print("\n" + "="*60)
        print("🤖 ДЕМО ОБУЧЕННЫХ МОДЕЛЕЙ ДЛЯ ГЕНЕРАЦИИ ТОКСИЧНОГО ТЕКСТА")
        print("="*60)
        print("⚠️  ВНИМАНИЕ: Модели обучены генерировать токсичный контент для исследовательских целей!")
        print("="*60)
        print("\nДоступные модели:")
        for key, model_info in self.models.items():
            print(f"  {key}. {model_info['name']}")
        print("\nИнструкции:")
        print("  1. Выберите номер модели (1 или 2)")
        print("  2. Дождитесь загрузки модели")
        print("  3. Введите промпт для генерации")
        print("  4. Получите ответ от модели")
        print("  5. Введите 'quit' для выхода")
        print("  6. Введите 'change' для смены модели")
        print("-"*60)

    def load_model(self, model_key: str) -> bool:
        """Load the selected model"""
        if model_key not in self.models:
            print(f"❌ Ошибка: Модель {model_key} не найдена")
            return False

        model_info = self.models[model_key]
        model_path = model_info["path"]
        model_type = model_info["type"]

        if not Path(model_path).exists():
            print(f"❌ Ошибка: Модель не найдена по пути {model_path}")
            print("Убедитесь, что модель была обучена и сохранена")
            return False

        try:
            print(f"\n🔄 Загружаю {model_info['name']}...")
            
            # Load tokenizer
            if model_type == "lora":
                # For LoRA models, load base model tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen3-0.6B", 
                    trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen3-0.6B",
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    device_map=self.device,
                    trust_remote_code=True
                )
                
                # Load LoRA adapter
                self.current_model = PeftModel.from_pretrained(base_model, model_path)
                print("✅ LoRA модель загружена успешно!")
                
            elif model_type == "rl":
                # For RL models, load directly
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.current_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    device_map=self.device,
                    trust_remote_code=True
                )
                print("✅ RL модель загружена успешно!")

            print(f"📱 Устройство: {self.device}")
            print(f"🎯 Текущая модель: {model_info['name']}")
            return True

        except Exception as e:
            print(f"❌ Ошибка при загрузке модели: {str(e)}")
            return False

    def generate_response(self, prompt: str) -> str:
        """Generate response for the given prompt"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.current_model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.current_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated text
            prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt")
            response_tokens = outputs[0][prompt_tokens.shape[1]:]
            generated_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            return generated_text.strip()

        except Exception as e:
            return f"❌ Ошибка генерации: {str(e)}"

    def run(self):
        """Run the interactive demo"""
        self.print_instructions()
        
        # Select initial model
        while True:
            choice = input("\n🎯 Выберите модель (1 или 2): ").strip()
            
            if choice in self.models:
                if self.load_model(choice):
                    break
            else:
                print("❌ Неверный выбор. Введите 1 или 2.")

        # Interactive loop
        print("\n🚀 Готов к работе! Введите промпт или команду:")
        print("  - 'quit' для выхода")
        print("  - 'change' для смены модели")
        print("  - любой текст для генерации ответа")
        print("-"*60)

        while True:
            user_input = input("\n💬 Введите промпт: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 До свидания!")
                break
                
            elif user_input.lower() == 'change':
                self.print_instructions()
                while True:
                    choice = input("\n🎯 Выберите модель (1 или 2): ").strip()
                    if choice in self.models:
                        if self.load_model(choice):
                            break
                    else:
                        print("❌ Неверный выбор. Введите 1 или 2.")
                continue
                
            elif not user_input:
                print("❌ Промпт не может быть пустым")
                continue
            
            # Generate response
            print("\n🤖 Генерирую ответ...")
            response = self.generate_response(user_input)
            
            print(f"\n📝 Ответ модели:")
            print(f"Промпт: {user_input}")
            print(f"Ответ: {response}")
            print("-"*60)


def main():
    """Main function"""
    try:
        demo = ModelDemo()
        demo.run()
    except KeyboardInterrupt:
        print("\n\n👋 Демо прервано пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {str(e)}")
        logger.error(f"Demo failed: {str(e)}")


if __name__ == "__main__":
    main()