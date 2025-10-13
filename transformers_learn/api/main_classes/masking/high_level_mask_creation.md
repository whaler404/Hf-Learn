# 高级掩码创建函数

本文档介绍用于创建各种类型注意力掩码的高级函数，这些函数封装了复杂的掩码创建逻辑。

## 目录
- [因果掩码创建](#因果掩码创建)
- [滑动窗口因果掩码创建](#滑动窗口因果掩码创建)
- [分块因果掩码创建](#分块因果掩码创建)
- [生成掩码创建](#生成掩码创建)
- [预处理器函数](#预处理器函数)

## 因果掩码创建

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | PretrainedConfig | 模型配置 |
| `input_embeds` | torch.Tensor | 输入嵌入，形状 `(batch_size, query_length, hidden_dim)` |
| `attention_mask` | Optional[torch.Tensor] | 2D 注意力掩码 |
| `cache_position` | torch.Tensor | 缓存位置索引 |
| `past_key_values` | Optional[Cache] | 过去的键值缓存 |
| `position_ids` | Optional[torch.Tensor] | 位置 ID |
| `or_mask_function` | Optional[Callable] | OR 组合的额外掩码函数 |
| `and_mask_function` | Optional[Callable] | AND 组合的额外掩码函数 |

### 实现逻辑

```python
def create_causal_mask(config, input_embeds, attention_mask, cache_position,
                      past_key_values=None, position_ids=None,
                      or_mask_function=None, and_mask_function=None):

    # 1. 处理混合缓存结构
    if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(False)
    else:
        layer_idx = 0

    # 2. 预处理掩码参数
    early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx
    )
    if early_exit:
        return attention_mask

    # 3. 设置基本参数
    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    mask_factory_function = causal_mask_function
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    # 4. 编译优化设置
    if _is_torch_xpu_available:
        allow_is_causal_skip = True
    else:
        allow_is_causal_skip = not getattr(past_key_values, "is_compileable", False)

    # 5. 应用掩码修饰函数
    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False

    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False

    # 6. 处理打包序列
    if packed_sequence_mask is not None and _is_torch_greater_or_equal_than_2_6:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False

    # 7. 创建最终掩码
    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,
        dtype=dtype,
        config=config,
    )

    return causal_mask
```

### 使用示例
```python
# 基础因果掩码
config = model.config
input_embeds = torch.randn(2, 10, 768)  # batch=2, seq_len=10, hidden_dim=768
cache_position = torch.arange(10)

mask = create_causal_mask(
    config=config,
    input_embeds=input_embeds,
    attention_mask=None,
    cache_position=cache_position,
    past_key_values=None
)

# 带图像 token 的掩码 (OR 模式)
def image_token_mask(batch_idx, head_idx, q_idx, kv_idx):
    # 允许文本 token 关注图像 token
    return q_idx < text_length and kv_idx >= image_start_pos

mask = create_causal_mask(
    config=config,
    input_embeds=input_embeds,
    attention_mask=None,
    cache_position=cache_position,
    past_key_values=None,
    or_mask_function=image_token_mask
)
```

## 滑动窗口因果掩码创建

### 特殊处理

与标准因果掩码不同，滑动窗口需要处理：

1. **滑动窗口大小**: 从配置中提取 `sliding_window` 参数
2. **混合缓存**: 如果存在混合缓存，选择滑动层
3. **局部大小传递**: 向底层掩码函数传递窗口大小

### 实现逻辑

```python
def create_sliding_window_causal_mask(config, input_embeds, attention_mask, cache_position,
                                     past_key_values=None, position_ids=None,
                                     or_mask_function=None, and_mask_function=None):

    # 1. 处理混合缓存结构 - 选择滑动层
    if hasattr(past_key_values, "is_sliding") and True in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(True)
    else:
        layer_idx = 0

    # 2. 预处理参数 (同因果掩码)
    early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx
    )
    if early_exit:
        return attention_mask

    # 3. 验证滑动窗口配置
    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        raise ValueError("Could not find a `sliding_window` argument in the config, or it is not set")

    # 4. 创建滑动窗口掩码函数
    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    mask_factory_function = sliding_window_causal_mask_function(sliding_window)
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    # 5. 其余逻辑同因果掩码创建
    # ... (允许跳过、掩码修饰、打包序列等处理)

    # 6. 创建掩码，传递 local_size 参数
    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,
        local_size=sliding_window,  # 关键：传递窗口大小
        dtype=dtype,
        config=config,
    )

    return causal_mask
```

### 滑动窗口可视化

对于 `sliding_window=3` 的序列长度 5：

```
q\kv 0 1 2 3 4
  0  ■ ⬚ ⬚ ⬚ ⬚
  1  ■ ■ ⬚ ⬚ ⬚
  2  ■ ■ ■ ⬚ ⬚
  3  ⬚ ■ ■ ■ ⬚
  4  ⬚ ⬚ ■ ■ ■
```

### 使用示例

```python
# Mistral 风格的滑动窗口掩码
config.sliding_window = 4096  # 4K 滑动窗口

mask = create_sliding_window_causal_mask(
    config=config,
    input_embeds=input_embeds,
    attention_mask=attention_mask,
    cache_position=cache_position,
    past_key_values=cache,
)

# 长序列的滑动窗口处理
long_seq_embeds = torch.randn(1, 8192, 768)  # 8K 序列
mask = create_sliding_window_causal_mask(
    config=config,
    input_embeds=long_seq_embeds,
    attention_mask=None,
    cache_position=torch.arange(8192),
    past_key_values=None
)
# 实际注意力范围: 每个位置只能看到前面 4096 个位置
```

## 分块因果掩码创建

### 特殊处理

1. **分块大小**: 从配置中提取 `attention_chunk_size`
2. **左填充处理**: 考虑序列开头的填充 token
3. **Flash Attention 限制**: 检查与 Flash Attention 的兼容性

### 实现逻辑

```python
def create_chunked_causal_mask(config, input_embeds, attention_mask, cache_position,
                              past_key_values=None, position_ids=None,
                              or_mask_function=None, and_mask_function=None):

    # 1. 处理混合缓存
    if hasattr(past_key_values, "is_sliding") and True in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(True)
    else:
        layer_idx = 0

    # 2. 预处理参数
    early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx
    )
    if early_exit:
        return attention_mask

    # 3. 验证分块配置
    chunk_size = getattr(config, "attention_chunk_size", None)
    if chunk_size is None:
        raise ValueError("Could not find an `attention_chunk_size` argument in the config, or it is not set")

    # 4. Flash Attention 兼容性检查
    if config._attn_implementation == "flash_attention_2" and kv_length + kv_offset > chunk_size:
        raise ValueError(
            "Flash attention 2 cannot handle chunked attention, and the key-value length is larger than the chunk size so the "
            "chunked pattern cannot be respected. You should use another `attn_implementation` when instantiating the model"
        )

    # 5. 计算左填充 token 数量
    batch_size = input_embeds.shape[0]
    if attention_mask is not None:
        left_padding_tokens = (attention_mask.cumsum(dim=-1) == torch.zeros_like(attention_mask)).sum(dim=-1)
    else:
        left_padding_tokens = torch.zeros(batch_size, device=cache_position.device, dtype=int)

    # 6. 旧版本 torch 警告
    if (not _is_torch_greater_or_equal_than_2_6 and
        kv_length + kv_offset > chunk_size and
        (left_padding_tokens > 0).any()):
        logger.warning_once(
            "Due to limitations of your current torch version, we cannot correctly account for the left-padding "
            "when computing the chunked attention pattern. This will lead to a wrong attention mask for the padded "
            "sequences. Behavior will be undefined. Please upgrade to `torch>=2.6` to solve this issue."
        )

    # 7. 创建分块掩码函数
    mask_factory_function = chunked_causal_mask_function(chunk_size, left_padding_tokens)
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    # 8. 其余处理逻辑
    # ... (掩码修饰、打包序列等)

    # 9. 创建掩码，传递 chunk_size 作为 local_size
    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,
        local_size=chunk_size,
        dtype=dtype,
        config=config,
    )

    return causal_mask
```

### 分块注意力可视化

对于 `chunk_size=4` 的序列长度 10：

```
q\kv 0 1 2 3 | 4 5 6 7 | 8 9
  0  ■ ■ ■ ⬚ | ⬚ ⬚ ⬚ ⬚ | ⬚ ⬚
  1  ■ ■ ■ ⬚ | ⬚ ⬚ ⬚ ⬚ | ⬚ ⬚
  2  ■ ■ ■ ⬚ | ⬚ ⬚ ⬚ ⬚ | ⬚ ⬚
  3  ■ ■ ■ ⬚ | ⬚ ⬚ ⬚ ⬚ | ⬚ ⬚
  ----------------+----------------+------
  4  ⬚ ⬚ ⬚ ⬚ | ■ ■ ■ ⬚ | ⬚ ⬚
  5  ⬚ ⬚ ⬚ ⬚ | ■ ■ ■ ⬚ | ⬚ ⬚
  6  ⬚ ⬚ ⬚ ⬚ | ■ ■ ■ ⬚ | ⬚ ⬚
  7  ⬚ ⬚ ⬚ ⬚ | ■ ■ ■ ⬚ | ⬚ ⬚
  ----------------+----------------+------
  8  ⬚ ⬚ ⬚ ⬚ | ⬚ ⬚ ⬚ ⬚ | ■ ■
  9  ⬚ ⬚ ⬚ ⬚ | ⬚ ⬚ ⬚ ⬚ | ■ ■
```

### 使用示例

```python
# Llama4 风格的分块注意力
config.attention_chunk_size = 2048  # 2K 分块

mask = create_chunked_causal_mask(
    config=config,
    input_embeds=input_embeds,
    attention_mask=attention_mask,
    cache_position=cache_position,
    past_key_values=cache,
)

# 处理填充序列
padded_attention_mask = torch.tensor([[True, True, True, False, False, True, True, True]])
mask = create_chunked_causal_mask(
    config=config,
    input_embeds=torch.randn(1, 8, 768),
    attention_mask=padded_attention_mask,
    cache_position=torch.arange(8),
    past_key_values=None
)
# 正确处理开头的填充 token
```

## 生成掩码创建

### 混合层处理

支持模型中不同层使用不同注意力模式：

```python
def create_masks_for_generate(config, input_embeds, attention_mask, cache_position,
                             past_key_values, position_ids=None, or_mask_function=None,
                             and_mask_function=None, **kwargs):

    # 1. 获取有效配置
    effective_config = config.get_text_config()

    # 2. 准备掩码参数
    mask_kwargs = {
        "config": effective_config,
        "input_embeds": input_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "or_mask_function": or_mask_function,
        "and_mask_function": and_mask_function,
    }

    # 3. 处理混合层模式
    if hasattr(effective_config, "layer_types"):
        causal_masks = {}
        for layer_pattern in set(effective_config.layer_types):
            mask_fn = LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING[layer_pattern]
            causal_masks[layer_pattern] = mask_fn(**mask_kwargs)
        return causal_masks

    # 4. 单一模式处理
    elif getattr(effective_config, "sliding_window", None) is not None:
        return create_sliding_window_causal_mask(**mask_kwargs)
    elif getattr(effective_config, "attention_chunk_size", None) is not None:
        return create_chunked_causal_mask(**mask_kwargs)
    else:
        return create_causal_mask(**mask_kwargs)
```

### 层模式映射

```python
LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING = {
    "full_attention": create_causal_mask,
    "sliding_attention": create_sliding_window_causal_mask,
    "chunked_attention": create_chunked_causal_mask,
}
```

### 使用示例

```python
# 混合层模型配置
config.layer_types = [
    "full_attention", "full_attention",      # 前 2 层：完全注意力
    "sliding_attention", "sliding_attention",  # 中间 2 层：滑动窗口
    "full_attention",                          # 最后 1 层：完全注意力
]
config.sliding_window = 1024

# 生成对应的掩码字典
masks = create_masks_for_generate(
    config=config,
    input_embeds=input_embeds,
    attention_mask=attention_mask,
    cache_position=cache_position,
    past_key_values=cache
)

# 结果: {'full_attention': mask1, 'sliding_attention': mask2}
```

## 预处理器函数

### _preprocess_mask_arguments

```python
def _preprocess_mask_arguments(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[Union[torch.Tensor, BlockMask]],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor],
    layer_idx: Optional[int],
) -> tuple[bool, Optional[Union[torch.Tensor, BlockMask]], Optional[torch.Tensor], int, int]
```

### 处理逻辑

1. **4D 掩码检查**: 如果已经是 4D 掩码，直接返回
2. **自定义实现检查**: 检查是否需要跳过掩码创建
3. **掩码预处理**: 移动设备、转换数据类型
4. **缓存信息提取**: 从缓存中获取长度和偏移信息
5. **打包序列检测**: 从 position_ids 检测打包序列

### 使用示例

```python
# 在自定义掩码创建中使用
early_exit, processed_mask, packed_mask, kv_len, kv_off = _preprocess_mask_arguments(
    config=config,
    input_embeds=input_embeds,
    attention_mask=attention_mask,
    cache_position=cache_position,
    past_key_values=cache,
    position_ids=position_ids,
    layer_idx=0
)

if early_exit:
    return processed_mask  # 直接返回预处理的掩码

# 继续创建自定义掩码...
```

## 最佳实践

### 1. 选择正确的掩码类型

```python
def choose_mask_type(seq_length, model_type, use_cache):
    if use_cache and seq_length > 4096:
        if model_type == "mistral":
            return "sliding_attention"
        elif model_type == "llama4":
            return "chunked_attention"
    return "full_attention"
```

### 2. 处理长序列

```python
def handle_long_sequences(config, input_embeds, attention_mask):
    seq_len = input_embeds.shape[1]

    if seq_len > config.max_position_embeddings:
        # 启用滑动窗口
        config.sliding_window = min(config.sliding_window or 4096, seq_len // 2)

        # 或启用分块注意力
        if hasattr(config, 'attention_chunk_size'):
            config.attention_chunk_size = min(config.attention_chunk_size, seq_len // 4)

    return create_masks_for_generate(config, input_embeds, attention_mask, ...)
```