# GenerationConfig 参数快速参考

## 参数速查表

### 长度控制
| 参数 | 类型 | 默认值 | 快速说明 |
|------|------|--------|----------|
| `max_length` | `int` | `20` | 最大总长度（含输入） |
| `max_new_tokens` | `int` | `None` | 最大新生成 token 数 |
| `min_length` | `int` | `0` | 最小总长度（含输入） |
| `min_new_tokens` | `int` | `None` | 最小新生成 token 数 |
| `early_stopping` | `bool/str` | `False` | Beam search 早停策略 |
| `max_time` | `float` | `None` | 最大生成时间（秒） |
| `stop_strings` | `str/list` | `None` | 停止字符串 |

### 生成策略
| 参数 | 类型 | 默认值 | 快速说明 |
|------|------|--------|----------|
| `do_sample` | `bool` | `False` | 是否使用采样 |
| `num_beams` | `int` | `1` | Beam search 宽度 |

### Logits 处理
| 参数 | 类型 | 默认值 | 常用值范围 | 快速说明 |
|------|------|--------|------------|----------|
| `temperature` | `float` | `1.0` | `0.1-2.0` | 控制随机性，越高越随机 |
| `top_k` | `int` | `50` | `1-100` | 只考虑前 K 个最可能的 token |
| `top_p` | `float` | `1.0` | `0.1-1.0` | Nucleus sampling，值越小越集中 |
| `min_p` | `float` | `None` | `0.01-0.2` | 最小概率阈值 |
| `typical_p` | `float` | `1.0` | `0.2-1.0` | 局部典型性采样 |
| `epsilon_cutoff` | `float` | `0.0` | `1e-4-1e-3` | 概率截断采样 |
| `eta_cutoff` | `float` | `0.0` | `1e-4-1e-3` | 混合截断采样 |
| `repetition_penalty` | `float` | `1.0` | `1.0-2.0` | 重复惩罚，>1.0 抑制重复 |
| `length_penalty` | `float` | `1.0` | `0.5-2.0` | 长度惩罚，>1.0 偏向长序列 |
| `no_repeat_ngram_size` | `int` | `0` | `2-4` | 禁止重复的 n-gram 大小 |

### 缓存控制
| 参数 | 类型 | 默认值 | 快速说明 |
|------|------|--------|----------|
| `use_cache` | `bool` | `True` | 是否使用 KV 缓存 |
| `cache_implementation` | `str` | `None` | 缓存实现类型 |
| `cache_config` | `dict` | `None` | 缓存配置参数 |
| `return_legacy_cache` | `bool` | `True` | 返回传统缓存格式 |

### 输出控制
| 参数 | 类型 | 默认值 | 快速说明 |
|------|------|--------|----------|
| `num_return_sequences` | `int` | `1` | 返回序列数量 |
| `output_attentions` | `bool` | `False` | 返回注意力权重 |
| `output_hidden_states` | `bool` | `False` | 返回隐藏状态 |
| `output_scores` | `bool` | `False` | 返回预测分数 |
| `output_logits` | `bool` | `None` | 返回原始 logits |
| `return_dict_in_generate` | `bool` | `False` | 返回完整输出字典 |

### 特殊标记
| 参数 | 类型 | 默认值 | 快速说明 |
|------|------|--------|----------|
| `pad_token_id` | `int` | `None` | 填充 token ID |
| `bos_token_id` | `int` | `None` | 开始 token ID |
| `eos_token_id` | `int/list` | `None` | 结束 token ID |

### 高级选项
| 参数 | 类型 | 默认值 | 快速说明 |
|------|------|--------|----------|
| `bad_words_ids` | `list[list[int]]` | `None` | 禁用的 token ID |
| `forced_bos_token_id` | `int` | `None` | 强制开始的 token ID |
| `forced_eos_token_id` | `int/list` | `None` | 强制结束的 token ID |
| `sequence_bias` | `dict` | `None` | 序列偏置 |
| `guidance_scale` | `float` | `None` | 分类器指导比例 |
| `watermarking_config` | `dict/config` | `None` | 水印配置 |

## 常用配置模板

### 1. 创意文本生成
```python
config = GenerationConfig(
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    max_new_tokens=200,
    repetition_penalty=1.1,
)
```

### 2. 事实性问答
```python
config = GenerationConfig(
    do_sample=False,  # 贪心解码
    max_new_tokens=100,
    num_beams=1,
    early_stopping=True,
)
```

### 3. Beam Search 高质量生成
```python
config = GenerationConfig(
    num_beams=5,
    max_new_tokens=150,
    early_stopping=True,
    no_repeat_ngram_size=2,
    length_penalty=1.2,
)
```

### 4. 低延迟生成
```python
config = GenerationConfig(
    do_sample=False,
    max_new_tokens=50,
    num_beams=1,
    use_cache=True,
)
```

### 5. 对话生成
```python
config = GenerationConfig(
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=100,
    repetition_penalty=1.05,
    no_repeat_ngram_size=3,
)
```

## 参数互斥性

某些参数组合是互斥的或需要特别注意：

1. **`do_sample=False` 时**，以下参数会被忽略：
   - `temperature`, `top_p`, `top_k`, `min_p`, `typical_p`, `epsilon_cutoff`, `eta_cutoff`

2. **`num_beams=1` 时**，以下参数会被忽略：
   - `early_stopping`, `length_penalty`, `num_beam_groups`

3. **`use_cache=False` 时**，缓存相关参数会被忽略：
   - `cache_implementation`, `cache_config`, `return_legacy_cache`

4. **`return_dict_in_generate=False` 时**，输出相关参数会被忽略：
   - `output_attentions`, `output_hidden_states`, `output_scores`, `output_logits`

## 性能提示

1. **提高速度**：
   - 使用 `use_cache=True`
   - 减小 `num_beams`
   - 使用贪心解码 (`do_sample=False`)

2. **提高质量**：
   - 使用 beam search (`num_beams>1`)
   - 添加重复惩罚 (`repetition_penalty>1.0`)
   - 使用 `no_repeat_ngram_size>0`

3. **控制多样性**：
   - 调整 `temperature`
   - 使用 `top_p` nucleus sampling
   - 组合 `top_k` 和 `top_p`

4. **内存优化**：
   - 限制 `max_new_tokens`
   - 使用适当的缓存实现
   - 考虑批处理大小