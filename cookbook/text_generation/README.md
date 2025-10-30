# 文本生成模型训练

模型： HuggingFaceTB/SmolLM2-135M
数据集： roneneldan/TinyStories
训练目标： 交叉熵损失，

使用库： datasets 、 transformers 、 peft 、 pytorch lightning

# 思路

## 数据集

使用 datasets 库加载数据集，数据集目录为
```
datasets/datasets--roneneldan--TinyStories
├── data
│   ├── train-00000-of-00004-2d5a1467fff1081b.parquet 
│   ├── train-00001-of-00004-5852b56a2bd28fd9.parquet 
│   ├── train-00002-of-00004-a26307300439e943.parquet 
│   ├── train-00003-of-00004-d243063613e5a057.parquet 
│   └── validation-00000-of-00001-869c898b519ad725.parquet 
```

数据集的列的 key 为 'text'

## 模型

模型架构为：
```python
{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "hidden_act": "silu",
  "hidden_size": 576,
  "initializer_range": 0.041666666666666664,
  "intermediate_size": 1536,
  "is_llama_config": true,
  "max_position_embeddings": 8192,
  "model_type": "llama",
  "num_attention_heads": 9,
  "num_hidden_layers": 30,
  "num_key_value_heads": 3,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_interleaved": false,
  "rope_scaling": null,
  "rope_theta": 100000,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.40.1",
  "use_cache": true,
  "vocab_size": 49152
}
```

从模型配置中获取有效信息
使用 transformer 加载模型

## 训练

使用 peft 的 prefix-tuning 对模型进行微调，使用 pytorch lightning 封装 peft 模型，构造训练流程

lightning 的 Hook 设置
- 日志：
train_loss
使用 TensorBoard 日志工具，保存在 trainer_output/text_generation/logs 中

- 检查点保存：性能最优时保存

模型保存在 trainer_output/text_generation 中，命名格式为 SmolLM2-135M-TinyStories-{epoch}-{train_loss}

- 早停：当验证指标长时间不提升时自动提前停止训练。

## 验证

最后跑一轮验证，同样在 pytorch lightning 中实现对应的步骤

val_loss

## 你需要生成三个文件

cookbook/text_generation
    ｜-- prepare_dataset.py # 定义对应的 dataloader
    ｜-- model.py # 定义 lightning 模型
    ｜-- main.py # 定义好训练配置

生成的代码应该简洁