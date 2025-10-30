import argparse
import os
import sys

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="将 PyTorch Lightning 的 .ckpt 转换为 PEFT 适配器格式")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

from model import TextGenerationModel

def main():
    args = parse_args()
    config = load_config(args.config)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


    ckpt_path = os.path.abspath(config["ckpt"])
    out_dir = os.path.abspath(config["out"])
    os.makedirs(out_dir, exist_ok=True)

    lm = TextGenerationModel.load_from_checkpoint(ckpt_path, map_location="cpu")

    peft_model = lm.model.to("cpu")
    peft_model.save_pretrained(out_dir)

    if config.get("save_tokenizer"):
        from transformers import AutoTokenizer

        base_model_name = getattr(lm.hparams, "model_name", None) or (
            lm.hparams.get("model_name") if isinstance(lm.hparams, dict) else None
        )

        if base_model_name:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
            tokenizer.save_pretrained(out_dir)

    # 生成简短的使用说明，写入到输出目录
    readme_path = os.path.join(out_dir, "README_conversion.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "用法说明\n"
            "1) 该目录保存的是 PEFT Prefix Tuning 适配器（非基础模型权重）。\n"
            "2) 由于 Prefix Tuning 无法像 LoRA 一样合并进基础模型，后续加载需使用 PEFT：\n\n"
            "Python 示例：\n"
            "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
            "from peft import PeftModel\n\n"
            "base = AutoModelForCausalLM.from_pretrained('<base_model_name>')\n"
            "tok = AutoTokenizer.from_pretrained('<base_model_name>')\n"
            "model = PeftModel.from_pretrained(base, r'{}')\n\n".format(out_dir).replace("\\", "\\\\")
            + "# 推理示例\n"
            + "model.eval()\n"
            + "inputs = tok('Hello', return_tensors='pt')\n"
            + "_ = model.generate(**inputs, max_new_tokens=20)\n"
        )

    print(f"转换完成：PEFT 适配器已保存到 {out_dir}")


if __name__ == "__main__":
    main()

