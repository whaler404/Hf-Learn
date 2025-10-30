#!/usr/bin/env python3

"""
Inference script for PEFT Prefix-Tuning model.

Usage:
    python cookbook/text_generation/inference.py --config config.yaml

YAML example:
    base_model: "HuggingFaceTB/SmolLM2-135M"
    peft_model_path: "path/to/peft_adapter"
    val_data_path: "path/to/validation.parquet"
    num_samples: 10
    max_new_tokens: 50
"""

import argparse
import yaml
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with PEFT model")
    parser.add_argument("--config", type=str, help="YAML config file path")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(peft_model_path):
    print(f"Loading PEFT model from: {peft_model_path}")

    # 直接加载PEFT模型
    model = AutoPeftModelForCausalLM.from_pretrained(peft_model_path).to("cuda")

    # 从PEFT目录加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(peft_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    return model, tokenizer


def load_validation_data(val_data_path):
    print(f"Loading validation data from: {val_data_path}")
    dataset = load_dataset("parquet", data_files=val_data_path, split="train")
    return dataset


def split_text_for_autoregressive(text):
    text = text.strip()
    mid_point = len(text) // 2
    while mid_point < len(text) and not text[mid_point].isspace():
        mid_point += 1

    prompt = text[:mid_point].strip()
    ground_truth = text[mid_point:].strip()

    return prompt, ground_truth


def generate_text(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):].strip()


def save_results_to_txt(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, result in enumerate(results, 1):
            f.write(f"=== Sample {i} ===\n")
            f.write(f"Prompt:\n{result['prompt']}\n\n")
            f.write(f"Model Generated:\n{result['generated']}\n\n")
            f.write(f"Ground Truth:\n{result['ground_truth']}\n\n")
            f.write("=" * 80 + "\n\n")


def main():
    args = parse_args()
    config = load_config(args.config)

    val_dataset = load_validation_data(config["val_data_path"])

    model, tokenizer = load_model_and_tokenizer(
        config["peft_model_path"]
    )

    num_samples = config.get("num_samples", 5)
    max_new_tokens = config.get("max_new_tokens", 100)
    output_file = config.get("output_file", "autoregressive_results.txt")

    print(f"\nRunning autoregressive generation for {num_samples} samples...")
    print(f"Results will be saved to: {output_file}")
    print("=" * 60)

    results = []

    for sample in val_dataset:
        original_text = sample["text"].strip()

        if len(original_text) < 100:
            continue

        prompt, ground_truth = split_text_for_autoregressive(original_text)

        if not (50 <= len(prompt) <= 512):
            continue

        print(f"\nProcessing sample {len(results) + 1}...")
        print(f"Prompt length: {len(prompt)} chars")

        generated = generate_text(model, tokenizer, prompt, max_new_tokens)

        results.append({
            'prompt': prompt,
            'generated': generated,
            'ground_truth': ground_truth
        })

        print(f"Generated {len(generated)} chars")

        if len(results) >= num_samples:
            break

    save_results_to_txt(results, output_file)
    print(f"\nCompleted! Generated {len(results)} samples.")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()