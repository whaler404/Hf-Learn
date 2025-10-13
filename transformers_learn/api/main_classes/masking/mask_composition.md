# Mask 组合函数

本文档介绍用于组合多个 mask 函数的高级函数，支持复杂的注意力模式构建。

## 目录
- [AND 掩码组合](#and-掩码组合)
- [OR 掩码组合](#or-掩码组合)
- [Mask 偏移函数](#mask-偏移函数)
- [实际应用示例](#实际应用示例)

## AND 掩码组合

返回多个 mask 函数的交集，即只有当所有 mask 函数都返回 True 时，注意力才被允许。

- `*mask_functions` (Callable): 可变数量的 mask 函数

```python
def and_masks(*mask_functions: Callable) -> Callable:
    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_ones((), dtype=torch.bool)
        for mask in mask_functions:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return and_mask
```

使用示例
```python
# 组合因果掩码和滑动窗口
causal_fn = causal_mask_function
sliding_fn = sliding_window_overlay(sliding_window=3)

# 创建组合掩码：既要满足因果性，又要在滑动窗口内
combined_fn = and_masks(causal_fn, sliding_fn)

# 生成完整 4D 掩码
mask = create_4d_mask(combined_fn, batch_size=2, seq_len=5)
# 最终形状: (2, num_heads, 5, 5)
```

可视化示例
因果 + 滑动窗口 (window=3):
```
原始因果:        滑动窗口:        组合结果:
■ ⬚ ⬚ ⬚ ⬚       ■ ⬚ ⬚ ⬚ ⬚       ■ ⬚ ⬚ ⬚ ⬚
■ ■ ⬚ ⬚ ⬚       ■ ■ ⬚ ⬚ ⬚       ■ ■ ⬚ ⬚ ⬚
■ ■ ■ ⬚ ⬚       ■ ■ ■ ⬚ ⬚       ■ ■ ■ ⬚ ⬚
■ ■ ■ ■ ⬚       ⬚ ■ ■ ■ ⬚       ⬚ ■ ■ ■ ⬚
■ ■ ■ ■ ■       ⬚ ⬚ ■ ■ ■       ⬚ ⬚ ■ ■ ■
```

## OR 掩码组合

返回多个 mask 函数的并集，只要任一 mask 函数返回 True，注意力就被允许。

- `*mask_functions` (Callable): 可变数量的 mask 函数

```python
def or_masks(*mask_functions: Callable) -> Callable:
    def or_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_zeros((), dtype=torch.bool)
        for mask in mask_functions:
            result = result | mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return or_mask
```

使用示例
```python
# 允许特定位置的全局注意力
def global_attention_mask_function(batch_idx, head_idx, q_idx, kv_idx):
    # 假设某些位置可以全局关注
    return kv_idx in global_positions[batch_idx]

# 组合：在滑动窗口内 OR 全局位置
combined_fn = or_masks(sliding_fn, global_attention_mask_function)
```

可视化示例
滑动窗口 OR 全局位置 (位置 2 可全局):
```
滑动窗口:        全局位置:        组合结果:
■ ⬚ ⬚ ⬚ ⬚       ⬚ ⬚ ■ ⬚ ⬚       ■ ⬚ ■ ⬚ ⬚
■ ■ ⬚ ⬚ ⬚       ⬚ ⬚ ■ ⬚ ⬚       ■ ■ ■ ⬚ ⬚
■ ■ ■ ⬚ ⬚       ⬚ ⬚ ■ ⬚ ⬚       ■ ■ ■ ⬚ ⬚
⬚ ■ ■ ■ ⬚       ⬚ ⬚ ■ ⬚ ⬚       ⬚ ■ ■ ■ ⬚
⬚ ⬚ ■ ■ ■       ⬚ ⬚ ■ ⬚ ⬚       ⬚ ⬚ ■ ■ ■
```

## Mask 偏移函数

为 mask 函数添加偏移量，用于处理缓存中的偏移索引。

参数
- `mask_function` (Callable): 原始 mask 函数
- `q_offset` (int): 查询位置的偏移量
- `kv_offset` (int): 键值位置的偏移量

```python
def add_offsets_to_mask_function(mask_function: Callable, q_offset: int, kv_offset: int) -> Callable:
    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return mask_function(batch_idx, head_idx, q_idx + q_offset, kv_idx + kv_offset)
    return inner_mask
```


使用示例
```python
# 基础因果掩码
base_causal = causal_mask_function

# 为缓存阶段添加偏移
offset_causal = add_offsets_to_mask_function(base_causal, q_offset=10, kv_offset=0)

# 处理增量生成时的掩码
current_pos = torch.tensor([10])  # 当前位置
past_length = 10  # 已处理的序列长度

mask = offset_causal(0, 0, 10, 5)  # 检查位置 10 是否可以关注位置 5
```

## 实际应用示例

示例 1: 混合注意力模式
```python
# 创建复杂的注意力模式：因果 + 滑动窗口 + 填充处理
def create_hybrid_mask(sliding_window, padding_mask):
    # 基础组件
    causal = causal_mask_function
    sliding = sliding_window_overlay(sliding_window)
    padding = padding_mask_function(padding_mask)

    # 组合所有约束
    return and_masks(causal, sliding, padding)

# 使用
padding_mask = torch.tensor([[True, True, True, False, False]])
hybrid_fn = create_hybrid_mask(sliding_window=3, padding_mask=padding_mask)
```

示例 2: 多模态注意力
```python
# 文本+图像的注意力模式
def multimodal_attention_mask(text_length, image_length):
    def text_text_mask(batch_idx, head_idx, q_idx, kv_idx):
        # 文本到文本：完全因果
        if q_idx < text_length and kv_idx < text_length:
            return kv_idx <= q_idx
        return False

    def image_image_mask(batch_idx, head_idx, q_idx, kv_idx):
        # 图像到图像：完全连接
        if q_idx >= text_length and kv_idx >= text_length:
            return True
        return False

    def text_image_mask(batch_idx, head_idx, q_idx, kv_idx):
        # 文本可以关注所有图像
        return q_idx < text_length and kv_idx >= text_length

    def image_text_mask(batch_idx, head_idx, q_idx, kv_idx):
        # 图像可以关注所有文本
        return q_idx >= text_length and kv_idx < text_length

    # 组合所有模式
    return or_masks(text_text_mask, image_image_mask, text_image_mask, image_text_mask)
```

示例 3: 分层缓存处理
```python
# 处理不同层的不同注意力模式
def layer_specific_mask(layer_idx, is_sliding=False, sliding_window=None):
    base_mask = causal_mask_function

    if is_sliding and sliding_window:
        sliding = sliding_window_overlay(sliding_window)
        base_mask = and_masks(base_mask, sliding)

    return base_mask

# 应用到不同层
layer_masks = {}
for layer_idx in range(num_layers):
    if layer_idx < 8:  # 前 8 层使用滑动窗口
        layer_masks[layer_idx] = layer_specific_mask(
            layer_idx, is_sliding=True, sliding_window=1024
        )
    else:  # 后 8 层使用完全注意力
        layer_masks[layer_idx] = layer_specific_mask(layer_idx)
```
