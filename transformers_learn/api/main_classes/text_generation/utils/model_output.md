# Transformer 生成模型输出详解

## 概述

Transformer 生成模型的输出通过 `ModelOutput` 类及其子类进行结构化管理。这些输出包含了生成过程中产生的所有重要信息，包括生成的序列、分数、注意力权重、隐藏状态等。

## 主要输出类

### 1. GenerateDecoderOnlyOutput - Decoder-only 模型输出

适用于仅包含解码器的生成模型（如 GPT、Llama 等），使用非束搜索方法时的输出。

```python
@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    """
    Decoder-only 生成模型的输出，使用非束搜索方法。

    Args:
        sequences (torch.LongTensor): 生成的序列
        scores (tuple[torch.FloatTensor]): 处理后的预测分数
        logits (tuple[torch.FloatTensor]): 未处理的预测分数
        attentions (tuple[tuple[torch.FloatTensor]]): 注意力权重
        hidden_states (tuple[tuple[torch.FloatTensor]]): 隐藏状态
        past_key_values (Cache): KV缓存
    """
```

#### 字段详解

| 字段名 | 数据类型 | 张量形状 | 说明 |
|--------|----------|----------|------|
| `sequences` | `torch.LongTensor` | `(batch_size, sequence_length)` | 最终生成的token序列 |
| `scores` | `tuple[torch.FloatTensor]` | `(max_new_tokens, batch_size, vocab_size)` | 每个生成步骤处理后的分数 |
| `logits` | `tuple[torch.FloatTensor]` | `(max_new_tokens, batch_size, vocab_size)` | 每个生成步骤的原始logits |
| `attentions` | `tuple[tuple[torch.FloatTensor]]` | `(max_new_tokens, num_layers, batch_size, num_heads, gen_len, seq_len)` | 注意力权重 |
| `hidden_states` | `tuple[tuple[torch.FloatTensor]]` | `(max_new_tokens, num_layers, batch_size, gen_len, hidden_size)` | 隐藏状态 |
| `past_key_values` | `Cache` | 变化 | KV缓存对象 |

#### 实际案例

假设使用 GPT-2 生成文本，输入为 "The weather is"，生成3个新token：

```python
# 输入
input_text = "The weather is"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids  # shape: [1, 4]

# 生成配置
generation_config = GenerationConfig(
    max_new_tokens=3,
    output_scores=True,
    output_logits=True,
    output_attentions=True,
    output_hidden_states=True,
    use_cache=True,
    return_dict_in_generate=True
)

# 模型输出
outputs = model.generate(
    input_ids=input_ids,
    generation_config=generation_config
)

# 输出结构分析
print("=== GenerateDecoderOnlyOutput 字段详解 ===")

# 1. sequences - 最终生成的序列
print(f"sequences shape: {outputs.sequences.shape}")
print(f"sequences content: {outputs.sequences}")
# 输出: torch.Size([1, 7]) - [1, 4] 输入 + [1, 3] 新生成
# 内容: [[464, 628, 373, 338, 1107, 284, 518]] (对应 "The weather is nice today")

# 2. scores - 处理后的分数（经过logits processor处理）
print(f"\nscores length: {len(outputs.scores)}")  # 3 个新生成的token
print(f"score[0] shape: {outputs.scores[0].shape}")  # 第一个新生token的分数
# 输出: torch.Size([1, 50257]) - [batch_size, vocab_size]
# 内容: 处理后的概率分数，已经过temperature、top_k/top_p等处理

# 3. logits - 原始预测分数
print(f"\nlogits length: {len(outputs.logits)}")
print(f"logits[0] shape: {outputs.logits[0].shape}")
# 输出: torch.Size([1, 50257]) - [batch_size, vocab_size]
# 内容: 模型输出的原始logits，未经过任何处理

# 4. attentions - 注意力权重
print(f"\nattentions length: {len(outputs.attentions)}")  # 3 个生成步骤
print(f"attentions[0] length: {len(outputs.attentions[0])}")  # 12 层（GPT-2 small）
print(f"attentions[0][0] shape: {outputs.attentions[0][0].shape}")
# 输出: torch.Size([1, 12, 1, 4]) - [batch_size, num_heads, query_len, key_len]
# query_len=1 因为只生成一个新token
# key_len=4 因为注意力整个输入序列 "The weather is"

# 5. hidden_states - 隐藏状态
print(f"\nhidden_states length: {len(outputs.hidden_states)}")  # 3 个生成步骤
print(f"hidden_states[0] length: {len(outputs.hidden_states[0])}")  # 12 层 + embedding
print(f"hidden_states[0][0] shape: {outputs.hidden_states[0][0].shape}")
# 输出: torch.Size([1, 1, 768]) - [batch_size, seq_len, hidden_size]
# 第一个张量是embedding层，后面12个是各层的隐藏状态

# 6. past_key_values - KV缓存
print(f"\npast_key_values type: {type(outputs.past_key_values)}")
if hasattr(outputs.past_key_values, 'seen_tokens'):
    print(f"seen_tokens: {outputs.past_key_values.seen_tokens}")
# 输出: DynamicCache 或其他缓存类型
# seen_tokens: 7 - 总共处理的token数量
```

### 2. GenerateEncoderDecoderOutput - Encoder-Decoder 模型输出

适用于编码器-解码器模型（如 T5、BART 等）的输出。

```python
@dataclass
class GenerateEncoderDecoderOutput(ModelOutput):
    """
    Encoder-Decoder 生成模型的输出。

    Args:
        sequences (torch.LongTensor): 生成的序列
        scores (tuple[torch.FloatTensor]): 处理后的预测分数
        encoder_attentions (tuple[torch.FloatTensor]): 编码器注意力权重
        encoder_hidden_states (tuple[torch.FloatTensor]): 编码器隐藏状态
        decoder_attentions (tuple[tuple[torch.FloatTensor]]): 解码器注意力权重
        decoder_hidden_states (tuple[tuple[torch.FloatTensor]]): 解码器隐藏状态
        past_key_values (Cache): KV缓存
    """
```

#### 字段详解

| 字段名 | 数据类型 | 张量形状 | 说明 |
|--------|----------|----------|------|
| `sequences` | `torch.LongTensor` | `(batch_size, decoder_sequence_length)` | 解码器生成的序列 |
| `scores` | `tuple[torch.FloatTensor]` | `(max_new_tokens, batch_size, vocab_size)` | 每个生成步骤的分数 |
| `encoder_attentions` | `tuple[torch.FloatTensor]` | `(num_layers, batch_size, num_heads, encoder_seq_len, encoder_seq_len)` | 编码器自注意力 |
| `encoder_hidden_states` | `tuple[torch.FloatTensor]` | `(num_layers, batch_size, encoder_seq_len, hidden_size)` | 编码器隐藏状态 |
| `decoder_attentions` | `tuple[tuple[torch.FloatTensor]]` | `(max_new_tokens, num_layers, batch_size, num_heads, decoder_len, encoder_len)` | 解码器注意力 |
| `decoder_hidden_states` | `tuple[tuple[torch.FloatTensor]]` | `(max_new_tokens, num_layers, batch_size, decoder_len, hidden_size)` | 解码器隐藏状态 |
| `past_key_values` | `Cache` | 变化 | 解码器KV缓存 |

#### 实际案例

使用 T5 模型进行翻译任务：

```python
# 输入：英文句子
input_text = "Translate English to French: The weather is nice today."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids  # shape: [1, 10]

# 生成配置
generation_config = GenerationConfig(
    max_new_tokens=5,
    output_scores=True,
    output_attentions=True,
    output_hidden_states=True,
    use_cache=True,
    return_dict_in_generate=True
)

# 模型输出
outputs = model.generate(
    input_ids=input_ids,
    generation_config=generation_config
)

# 输出结构分析
print("=== GenerateEncoderDecoderOutput 字段详解 ===")

# 1. sequences - 生成的法文翻译
print(f"sequences shape: {outputs.sequences.shape}")
print(f"sequences: {outputs.sequences}")
# 输出: torch.Size([1, 5]) - 只包含解码器生成的部分
# 内容: [[150, 139, 284, 87, 1]] (对应 "Il fait beau")

# 2. scores - 解码器每个步骤的分数
print(f"\nscores length: {len(outputs.scores)}")  # 5 个token
print(f"scores[0] shape: {outputs.scores[0].shape}")
# 输出: torch.Size([1, 32128]) - [batch_size, vocab_size]

# 3. encoder_attentions - 编码器注意力（只计算一次）
print(f"\nencoder_attentions length: {len(outputs.encoder_attentions)}")  # 6 层（T5-small）
print(f"encoder_attentions[0] shape: {outputs.encoder_attentions[0].shape}")
# 输出: torch.Size([1, 8, 10, 10]) - [batch_size, num_heads, seq_len, seq_len]
# 编码器对输入句子的自注意力

# 4. encoder_hidden_states - 编码器隐藏状态
print(f"\nencoder_hidden_states length: {len(outputs.encoder_hidden_states)}")
print(f"encoder_hidden_states[0] shape: {outputs.encoder_hidden_states[0].shape}")
# 输出: torch.Size([1, 10, 512]) - [batch_size, seq_len, hidden_size]

# 5. decoder_attentions - 解码器注意力（交叉注意力）
print(f"\ndecoder_attentions length: {len(outputs.decoder_attentions)}")  # 5 个生成步骤
print(f"decoder_attentions[0] length: {len(outputs.decoder_attentions[0])}")  # 6 层
print(f"decoder_attentions[0][0] shape: {outputs.decoder_attentions[0][0].shape}")
# 输出: torch.Size([1, 8, 1, 10]) - [batch_size, num_heads, decoder_seq_len, encoder_seq_len]
# 解码器对编码器输出的交叉注意力

# 6. decoder_hidden_states - 解码器隐藏状态
print(f"\ndecoder_hidden_states length: {len(outputs.decoder_hidden_states)}")
print(f"decoder_hidden_states[0] length: {len(outputs.decoder_hidden_states[0])}")  # 6 层
print(f"decoder_hidden_states[0][0] shape: {outputs.decoder_hidden_states[0][0].shape}")
# 输出: torch.Size([1, 1, 512]) - [batch_size, seq_len, hidden_size]
```

### 3. GenerateBeamDecoderOnlyOutput - Beam Search 输出

适用于使用束搜索的 decoder-only 模型输出。

```python
@dataclass
class GenerateBeamDecoderOnlyOutput(ModelOutput):
    """
    使用束搜索的 decoder-only 生成模型输出。

    Args:
        sequences (torch.LongTensor): 生成的序列
        sequences_scores (torch.FloatTensor): 序列分数
        scores (tuple[torch.FloatTensor]): token分数
        logits (tuple[torch.FloatTensor]): 原始logits
        beam_indices (torch.LongTensor): 束索引
        attentions (tuple[tuple[torch.FloatTensor]]): 注意力权重
        hidden_states (tuple[tuple[torch.FloatTensor]]): 隐藏状态
        past_key_values (Cache): KV缓存
    """
```

#### 特有字段详解

| 字段名 | 数据类型 | 张量形状 | 说明 |
|--------|----------|----------|------|
| `sequences_scores` | `torch.FloatTensor` | `(batch_size, num_beams)` | 每个beam的序列分数 |
| `beam_indices` | `torch.LongTensor` | `(batch_size, num_beams, sequence_length)` | 束选择索引 |

#### 束搜索案例

```python
# 使用束搜索生成
generation_config = GenerationConfig(
    max_new_tokens=3,
    num_beams=3,
    output_scores=True,
    return_dict_in_generate=True
)

outputs = model.generate(
    input_ids=input_ids,
    generation_config=generation_config
)

print("=== GenerateBeamDecoderOnlyOutput 特有字段 ===")

# 1. sequences - 包含多个beam的序列
print(f"sequences shape: {outputs.sequences.shape}")
# 输出: torch.Size([3, 7]) - [num_beams, sequence_length]
# 内容: 3个不同的生成候选

# 2. sequences_scores - 每个beam的对数概率分数
print(f"sequences_scores shape: {outputs.sequences_scores.shape}")
print(f"sequences_scores: {outputs.sequences_scores}")
# 输出: torch.Size([3])
# 内容: [-2.34, -1.87, -2.56] - 3个beam的分数，越小越好

# 3. beam_indices - 束选择路径
print(f"beam_indices shape: {outputs.beam_indices.shape}")
print(f"beam_indices: {outputs.beam_indices}")
# 输出: torch.Size([3, 7]) - [num_beams, sequence_length]
# 内容: [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 2, 2], [0, 0, 0, 1, 0, 0, 0]]
# 表示每个位置选择的是哪个beam的路径
```

## 输出字段的典型应用场景

### 1. 序列分析
```python
# 获取最佳序列
best_sequence = outputs.sequences[0]  # 第一个序列通常是最佳序列
best_text = tokenizer.decode(best_sequence, skip_special_tokens=True)

# 束搜索时获取所有候选
if hasattr(outputs, 'sequences_scores'):
    for i, (seq, score) in enumerate(zip(outputs.sequences, outputs.sequences_scores)):
        text = tokenizer.decode(seq, skip_special_tokens=True)
        print(f"Candidate {i}: {text} (score: {score:.2f})")
```

### 2. 分数分析
```python
# 分析token概率分布
if outputs.scores:
    for step_idx, step_scores in enumerate(outputs.scores):
        top_tokens = torch.topk(step_scores[0], k=5)
        print(f"Step {step_idx + 1} top tokens:")
        for token_id, score in zip(top_tokens.indices, top_tokens.values):
            token = tokenizer.decode(token_id)
            print(f"  {token}: {score:.4f}")
```

### 3. 注意力可视化
```python
# 可视化注意力权重
if outputs.attentions:
    # 取最后一层的注意力
    last_attention = outputs.attentions[-1][-1]  # 最后一个步骤，最后一层
    attention_matrix = last_attention[0].mean(dim=0).cpu().numpy()  # 平均所有头

    import matplotlib.pyplot as plt
    plt.imshow(attention_matrix, cmap='Blues')
    plt.title("Attention Weights")
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    plt.colorbar()
    plt.show()
```

### 4. 隐藏状态分析
```python
# 分析隐藏状态的相似性
if outputs.hidden_states:
    # 取最后一层的隐藏状态
    last_hidden = outputs.hidden_states[-1][-1]  # 最后一个步骤，最后一层

    # 计算token间的相似度
    similarity = torch.cosine_similarity(
        last_hidden[0, 0:1], last_hidden[0], dim=1
    )
    print("Token similarities to first token:", similarity)
```