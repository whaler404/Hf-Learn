#!/usr/bin/env python3

"""
Utility to create a subset from existing datasets using YAML configuration.

Usage:
    python cookbook/utils/sample_dataset.py --config config.yaml

YAML example:
    data_files: ["file1.parquet", "file2.parquet"]
    select_num: 200
    save_dir: "output_dir"
"""

import argparse
import os
import yaml
from datasets import Dataset, DatasetDict, load_dataset

RNG_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Create dataset subset from YAML config")
    parser.add_argument("--config", type=str, required=True, help="YAML config file path")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_data_files(data_files):
    if isinstance(data_files, str):
        return [data_files]
    return list(data_files)


def pick_dataset(dataset_obj):
    if isinstance(dataset_obj, DatasetDict):
        return dataset_obj.get("train") or next(iter(dataset_obj.values()))
    return dataset_obj


def select_subset(dataset, limit):
    limit = min(max(int(limit), 1), dataset.num_rows)
    return dataset if limit == dataset.num_rows else dataset.select(range(limit))


def smart_split(dataset):
    n = dataset.num_rows
    if n < 3:
        return DatasetDict({"train": dataset})

    test_size = max(1, int(n * 0.2))
    train_valid = dataset.train_test_split(test_size=test_size, seed=RNG_SEED)
    test = train_valid["test"]
    train_candidate = train_valid["train"]

    m = train_candidate.num_rows
    if m < 2:
        return DatasetDict({"train": train_candidate, "test": test})

    valid_size = max(1, int(m * 0.1))
    split2 = train_candidate.train_test_split(test_size=valid_size, seed=RNG_SEED)

    return DatasetDict({
        "train": split2["train"],
        "validation": split2["test"],
        "test": test
    })


def save_splits(splits, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    filenames = {"train": "train.parquet", "validation": "validation.parquet", "test": "test.parquet"}

    for name, dataset in splits.items():
        output_path = os.path.join(save_dir, filenames.get(name, f"{name}.parquet"))
        dataset.to_parquet(output_path)
        print(f"Saved {dataset.num_rows:>4} rows to {output_path}")


def main():
    args = parse_args()
    config = load_config(args.config)

    data_files = normalize_data_files(config["data_files"])
    select_num = int(config.get("select_num", 200))
    save_dir = config.get("save_dir", "datasets/subset")

    print(f"Loading dataset from {len(data_files)} file(s)...")
    dataset_obj = load_dataset("parquet", data_files=data_files)
    dataset = pick_dataset(dataset_obj)
    print(f"Source dataset rows: {dataset.num_rows}")

    subset = select_subset(dataset, select_num)
    print(f"Selected subset rows: {subset.num_rows}")

    splits = smart_split(subset)
    for name, ds in splits.items():
        print(f"    {name}: {ds.num_rows} rows")

    save_splits(splits, save_dir)
    print(f"Done. Files saved under: {save_dir}")


if __name__ == "__main__":
    main()
