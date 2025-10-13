# 基础掩码函数

本文档介绍构成 Transformer 注意力机制基础的基本掩码函数。

## 目录
- [因果掩码函数](#因果掩码函数)
- [滑动窗口叠加](#滑动窗口叠加)
- [分块叠加](#分块叠加)
- [填充掩码函数](#填充掩码函数)
- [打包序列掩码函数](#打包序列掩码函数)

## 因果掩码函数

```python
def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return kv_idx <= q_idx
```

创建基础的下三角因果掩码，确保每个 token 只能关注自己和序列中前面的 token。

该函数实现简单规则：位置 `q_idx` 的查询可以关注位置 `kv_idx` 的键值，当且仅当 `kv_idx <= q_idx`。

对于长度为 5 的序列，因果掩码模式：
```
q\kv 0 1 2 3 4
  0  ■ ⬚ ⬚ ⬚ ⬚
  1  ■ ■ ⬚ ⬚ ⬚
  2  ■ ■ ■ ⬚ ⬚
  3  ■ ■ ■ ■ ⬚
  4  ■ ■ ■ ■ ■
```
■ = True (允许), ⬚ = False (阻塞)


## 滑动窗口叠加

```python
def sliding_window_overlay(sliding_window: int) -> Callable:
    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return kv_idx > q_idx - sliding_window
    return inner_mask
```

创建滑动窗口注意力模式，将注意力限制在每个查询位置周围的局部邻域。

内部函数允许注意力，当键值位置在查询位置后方 `sliding_window` 个 token 范围内时。

对于序列长度 5 且 sliding_window=3 的情况：
```
q\kv 0 1 2 3 4
  0  ■ ⬚ ⬚ ⬚ ⬚
  1  ■ ■ ⬚ ⬚ ⬚
  2  ■ ■ ■ ⬚ ⬚
  3  ⬚ ■ ■ ■ ⬚
  4  ⬚ ⬚ ■ ■ ■
```

## 分块叠加

创建分块注意力模式，其中 token 只能在其块内进行注意力。

- `chunk_size` (int): 每个块的大小
- `left_padding` (torch.Tensor): 表示每个批次中填充 token 数量的张量

```python
def chunked_overlay(chunk_size: int, left_padding: torch.Tensor) -> Callable:
    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return (kv_idx - left_padding[batch_idx]) // chunk_size == (q_idx - left_padding[batch_idx]) // chunk_size
    return inner_mask
```

该函数计算查询和键值位置的块索引，只允许在同一块内进行注意力。

对于序列长度 6 且 chunk_size=3 的情况：
```
q\kv 0 1 2 3 4 5
  0  ■ ■ ■ ⬚ ⬚ ⬚
  1  ■ ■ ■ ⬚ ⬚ ⬚
  2  ■ ■ ■ ⬚ ⬚ ⬚
  3  ⬚ ⬚ ⬚ ■ ■ ■
  4  ⬚ ⬚ ⬚ ■ ■ ■
  5  ⬚ ⬚ ⬚ ■ ■ ■
```

使用示例
```python
# 创建分块掩码
left_padding = torch.zeros(batch_size, dtype=int)
chunked_mask = chunked_overlay(chunk_size=4, left_padding=left_padding)
full_mask = and_masks(chunked_mask, causal_mask_function)
```

## 填充掩码函数

创建处理输入序列中填充 token 的掩码函数。

```python
def padding_mask_function(padding_mask: torch.Tensor) -> Callable:
    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return padding_mask[batch_idx, kv_idx]
    return inner_mask
```

该函数检查键值位置是否对应有效（非填充）的 token。

- 输入 `padding_mask` 形状：`(batch_size, seq_len)`
- 最终掩码形状：`(batch_size, num_heads, seq_len, seq_len)`

对于包含填充的批次（P = 填充 token）：
```
批次 0: [T, T, T, P, P]
q_idx=2 的掩码模式:
kv_idx: 0 1 2 3 4
       ■ ■ ■ ⬚ ⬚
```

使用示例
```python
# 创建填充掩码
padding_mask = torch.tensor([[True, True, True, False, False]])
padding_fn = padding_mask_function(padding_mask)
full_mask = and_masks(padding_fn, causal_mask_function)
```

## 打包序列掩码函数

处理打包序列，其中多个序列连接在单个批次维度中。

```python
def packed_sequence_mask_function(packed_sequence_mask: torch.Tensor) -> Callable:
    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
    return inner_mask
```

只允许属于同一打包序列的 token 之间进行注意力。

- 输入形状：`(batch_size, seq_len)` ，相等值表示相同序列的张量
- 最终掩码形状：`(batch_size, num_heads, seq_len, seq_len)`

### 使用示例
```python
# 从 position_ids 检测打包序列
packed_mask = find_packed_sequence_indices(position_ids)
packed_fn = packed_sequence_mask_function(packed_mask)
```