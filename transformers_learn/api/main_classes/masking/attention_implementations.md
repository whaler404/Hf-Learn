# 注意力实现掩码

本文档详细介绍不同注意力后端（SDPA、Eager、Flash Attention、Flex Attention）的掩码创建函数。

## 目录
- [填充掩码预处理](#填充掩码预处理)
- [SDPA 掩码](#sdpa-掩码)
- [Eager 掩码](#eager-掩码)
- [Flash Attention 掩码](#flash-attention-掩码)
- [Flex Attention 掩码](#flex-attention-掩码)
- [掩码接口选择](#掩码接口选择)

### 填充掩码预处理

在创建掩码之前，需要先处理填充掩码。`prepare_padding_mask` 函数负责准备正确的填充掩码：

#### 函数签名
```python
def prepare_padding_mask(
    attention_mask: Optional[torch.Tensor],
    kv_length: int,
    kv_offset: int,
    _slice: bool = True
) -> Optional[torch.Tensor]
```

- `attention_mask` (Optional[torch.Tensor]): 2D 注意力掩码，True 表示有效 token
- `kv_length` (int): 键值序列长度
- `kv_offset` (int): 键值序列的偏移量
- `_slice` (bool): 是否进行切片操作，Flex Attention 设为 False

#### 实现逻辑
```python
def prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=True):
    local_padding_mask = attention_mask
    if attention_mask is not None:
        # 1. 必要时进行填充
        if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0:
            local_padding_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))

        # 2. 根据偏移量切片 (Flex Attention 除外)
        if _slice:
            # 避免数据依赖切片 (torch.compile 友好)
            mask_indices = torch.arange(kv_length, device=local_padding_mask.device)
            mask_indices += kv_offset
            local_padding_mask = local_padding_mask[:, mask_indices]

    return local_padding_mask
```

#### 处理步骤详解

**步骤 1: 填充检查**
```python
# 计算需要填充的长度
padding_length = kv_length + kv_offset - attention_mask.shape[-1]

# 示例:
# attention_mask.shape[-1] = 8 (原始掩码长度)
# kv_length = 5, kv_offset = 3
# padding_length = 5 + 3 - 8 = 0 (无需填充)

# 如果:
# attention_mask.shape[-1] = 5
# kv_length = 5, kv_offset = 3
# padding_length = 5 + 3 - 5 = 3 (需要填充 3 个位置)
```

**步骤 2: 必要时填充**
```python
if padding_length > 0:
    # 在右侧填充 False (表示填充 token)
    local_padding_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))

# 示例:
# 原始: [True, True, True, True, True]
# 填充后: [True, True, True, True, True, False, False, False]
```

**步骤 3: 根据偏移量切片**
```python
# 创建目标索引 (避免数据依赖切片)
mask_indices = torch.arange(kv_length, device=local_padding_mask.device)  # [0, 1, 2, 3, 4]
mask_indices += kv_offset  # [3, 4, 5, 6, 7]

# 使用高级索引切片 (torch.compile 友好)
local_padding_mask = local_padding_mask[:, mask_indices]

# 示例:
# 原始掩码: [T, T, T, T, T, F, F, F]
# kv_offset=3, kv_length=5
# 提取位置: [3, 4, 5, 6, 7]
# 结果: [T, T, F, F, F]
```

#### 张量形状变化示例

```python
# 示例场景
attention_mask = torch.tensor([[True, True, True, True, True]])  # (1, 5)
kv_length = 4
kv_offset = 2

# 步骤 1: 检查填充
# padding_length = 4 + 2 - 5 = 1 (需要填充 1 个)

# 步骤 2: 填充
# attention_mask -> [[True, True, True, True, True, False]]  # (1, 6)

# 步骤 3: 切片
# mask_indices = [2, 3, 4, 5]
# result -> [[True, True, False, False]]  # (1, 4)
```

#### 使用场景

**1. 生成阶段 (带缓存)**
```python
# 编码阶段: 完整序列
encoder_mask = torch.tensor([[True, True, True, True, True]])
# 生成阶段: 只处理新 token
cache_position = torch.tensor([5])  # 新 token 位置
kv_length = 6  # 总 KV 长度
kv_offset = 0   # KV 从 0 开始

prepared_mask = prepare_padding_mask(encoder_mask, kv_length, kv_offset)
# 结果: [[True, True, True, True, True, False]] (填充到长度 6)
```

**2. 滑动窗口注意力**
```python
# 长序列 + 滑动窗口
attention_mask = torch.tensor([[True, True, True, True, True, True, True, True]])
kv_length = 4      # 窗口大小
kv_offset = 3      # 窗口起始位置

prepared_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)
# 提取位置 [3, 4, 5, 6] 的掩码值
# 结果: [[True, True, True, True]]
```

**3. Flex Attention 特殊处理**
```python
# Flex Attention 不切片，使用偏移量
prepared_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)

# 掩码函数会通过 add_offsets_to_mask_function 处理偏移
mask_function = add_offsets_to_mask_function(base_mask, q_offset, kv_offset)
```


## SDPA 掩码

SDPA（Scaled Dot-Product Attention）是 PyTorch 的标准注意力实现，支持高度优化的注意力计算。

### 函数签名
新版本 PyTorch (>=2.6)
```python
def sdpa_mask_recent_torch(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    **kwargs,
) -> Optional[torch.Tensor]
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `batch_size` | int | 批次大小 |
| `cache_position` | torch.Tensor | 查询位置的索引张量，形状 `(query_length,)` |
| `kv_length` | int | 键值序列长度 |
| `kv_offset` | int | 键值序列的偏移量 |
| `mask_function` | Callable | 掩码生成函数 |
| `attention_mask` | Optional[torch.Tensor] | 2D 填充掩码 |
| `local_size` | Optional[int] | 局部注意力窗口大小 |
| `allow_is_causal_skip` | bool | 是否允许使用 `is_causal` 参数跳过掩码创建 |

### 实现逻辑

```python
def sdpa_mask_recent_torch(batch_size, cache_position, kv_length, kv_offset=0,
                          mask_function=causal_mask_function, attention_mask=None,
                          local_size=None, allow_is_causal_skip=True, **kwargs):

    # cache_position: (q_length,) -> [0, 1, 2, ..., q_length-1]
    q_length = cache_position.shape[0]

    # 1. 准备填充掩码
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)

    # 2. 检查是否可以跳过掩码创建
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None

    # 3. 创建键值索引
    kv_arange = torch.arange(kv_length, device=cache_position.device)
    kv_arange += kv_offset
    # kv_arange: (kv_length,) -> [kv_offset, kv_offset+1, ..., kv_offset+kv_length-1]

    # 4. 组合填充掩码
    if padding_mask is not None:
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    # 5. 使用 vmap 创建 4D 掩码
    batch_arange = torch.arange(batch_size, device=cache_position.device)
    # batch_arange: (batch_size,) -> [0, 1, 2, ..., batch_size-1]
    head_arange = torch.arange(1, device=cache_position.device)
    # head_arange: (1,) -> [0]

    with TransformGetItemToIndex():
        causal_mask = _vmap_for_bhqkv(mask_function)(batch_arange, head_arange, cache_position, kv_arange)

    return causal_mask # (batch_size, 1, q_length, kv_length)
```

### 掩码跳过条件

SDPA 可以在某些条件下跳过掩码创建，使用原生 `is_causal` 参数：

```python
def _ignore_causal_mask_sdpa(padding_mask, query_length, kv_length, kv_offset, local_size):
    # 跳过条件：
    # 1. 非追踪模式
    # 2. query_length == 1 或 kv_length == query_length
    # 3. 无特殊局部注意力模式
    # 4. 无填充或全部有效

    is_tracing = torch.jit.is_tracing() or isinstance(padding_mask, torch.fx.Proxy) or is_torchdynamo_compiling()

    if (not is_tracing and
        (query_length == 1 or kv_length == query_length) and
        (local_size is None or kv_length < local_size) and
        (padding_mask is None or padding_mask.all())):
        return True
    return False
```

### 使用示例

```python
# 基础因果掩码
cache_pos = torch.arange(5)  # [0, 1, 2, 3, 4]
mask = sdpa_mask(
    batch_size=2,
    cache_position=cache_pos,
    kv_length=5,
    mask_function=causal_mask_function
)
# 0 ■ ⬚ ⬚ ⬚ ⬚
# 1 ■ ■ ⬚ ⬚ ⬚
# 2 ■ ■ ■ ⬚ ⬚
# 3 ■ ■ ■ ■ ⬚
# 4 ■ ■ ■ ■ ■
# mask shape: (2, 1, 5, 5)

# 滑动窗口掩码
sliding_fn = sliding_window_causal_mask_function(sliding_window=3)
mask = sdpa_mask(
    batch_size=1,
    cache_position=torch.arange(5),
    kv_length=10,
    kv_offset=5,
    mask_function=sliding_fn,
    local_size=3
)
# 0 ■ ⬚ ⬚ ⬚ ⬚
# 1 ■ ■ ⬚ ⬚ ⬚
# 2 ■ ■ ■ ⬚ ⬚
# 3 ⬚ ■ ■ ■ ⬚
# 4 ⬚ ⬚ ■ ■ ■

# 分块注意力掩码
chunked_causal_fn = chunked_causal_mask_function(3, torch.zeros(1, dtype=int))
sdpa_mask(
    batch_size=1, 
    cache_position=torch.arange(5), 
    kv_length=5, 
    mask_function=chunked_causal_fn
)
# 0 ■ ⬚ ⬚ ⬚ ⬚
# 1 ■ ■ ⬚ ⬚ ⬚
# 2 ■ ■ ■ ⬚ ⬚
# 3 ⬚ ⬚ ⬚ ■ ⬚
# 4 ⬚ ⬚ ⬚ ■ ■
```

## Eager 掩码

Eager 掩码用于显式的注意力计算，返回浮点数掩码（0 和 -inf）。

### 函数签名
```python
def eager_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> torch.Tensor
```

### 实现逻辑

```python
def eager_mask(batch_size, cache_position, kv_length, kv_offset=0,
               mask_function=causal_mask_function, attention_mask=None,
               dtype=torch.float32, **kwargs):

    # 1. 获取布尔掩码
    mask = sdpa_mask(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=False,  # 不允许跳过
        allow_torch_fix=False,
        **kwargs,
    )

    # 2. 转换为浮点数掩码
    min_dtype = torch.finfo(dtype).min
    mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)

    return mask
```

- `True` -> `0.0` (允许注意力)
- `False` -> `min_dtype` (阻塞注意力)

### 使用示例

```python
# 创建用于显式注意力的掩码
mask = eager_mask(
    batch_size=1,
    cache_position=torch.arange(3),
    kv_length=5,
    dtype=torch.float16
)

# 在注意力计算中使用
# attention_scores = torch.matmul(query, key.transpose(-2, -1))
# attention_scores = attention_scores + mask
# attention_weights = F.softmax(attention_scores, dim=-1)
```

## Flash Attention 掩码

Flash Attention 是高度优化的注意力实现，需要特殊的掩码格式。

### 函数签名
```python
def flash_attention_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Optional[torch.Tensor]
```

### 实现逻辑

```python
def flash_attention_mask(batch_size, cache_position, kv_length, kv_offset=0,
                        mask_function=causal_mask_function, attention_mask=None, **kwargs):

    if attention_mask is not None:
        # 1. 切片获取相关部分
        attention_mask = attention_mask[:, -kv_length:]

        # 2. 检查是否全为有效 token
        if attention_mask.all():
            attention_mask = None

    return attention_mask
```

### 特点

1. **无填充掩码**: Flash Attention 天然不支持填充
2. **返回 2D 掩码**: 用于提取序列长度
3. **自动优化**: 完全因果时返回 None

### 使用示例

```python
# Flash Attention 掩码
attention_mask = torch.tensor([[True, True, True, False, False]])
mask = flash_attention_mask(
    batch_size=1,
    cache_position=torch.arange(3),
    kv_length=3,
    attention_mask=attention_mask
)

# 如果 attention_mask 全为 True，返回 None
# 否则返回切片后的 2D 掩码
```

## Flex Attention 掩码

Flex Attention 是 PyTorch 最新的灵活注意力实现，使用 BlockMask 压缩表示。

### 函数签名
```python
def flex_attention_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> BlockMask
```

### 实现逻辑

```python
def flex_attention_mask(batch_size, cache_position, kv_length, kv_offset=0,
                       mask_function=causal_mask_function, attention_mask=None, **kwargs):

    q_length, q_offset = cache_position.shape[0], cache_position[0]

    # 1. 处理填充掩码
    if attention_mask is not None:
        # 兼容旧版本 torch 的块大小要求
        if not _is_torch_greater_or_equal_than_2_6:
            pad_len = ((attention_mask.shape[1] // flex_default_block_size) + 1) * flex_default_block_size
            pad_len = pad_len - attention_mask.shape[1]
            if pad_len > 0:
                attention_mask = torch.nn.functional.pad(attention_mask, value=0, pad=(0, pad_len))

        padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    # 2. 添加偏移量
    mask_function = add_offsets_to_mask_function(mask_function, q_offset, kv_offset)

    # 3. 创建 BlockMask
    block_mask = create_block_mask(
        mask_mod=mask_function,
        B=batch_size,
        H=None,
        Q_LEN=q_length,
        KV_LEN=kv_length,
        device=cache_position.device,
        _compile=_is_torch_greater_or_equal_than_2_6,
    )

    return block_mask
```

### BlockMask 压缩

BlockMask 将稀疏的注意力模式压缩为块级表示：

```
完整掩码 (16x16):
■ ⬚ ⬚ ⬚ | ⬚ ⬚ ⬚ ⬚ | ⬚ ⬚ ⬚ ⬚ | ⬚ ⬚ ⬚ ⬚
■ ■ ⬚ ⬚ | ⬚ ⬚ ⬚ ⬚ | ⬚ ⬚ ⬚ ⬚ | ⬚ ⬚ ⬚ ⬚
■ ■ ■ ⬚ | ⬚ ⬚ ⬚ ⬚ | ⬚ ⬚ ⬚ ⬚ | ⬚ ⬚ ⬚ ⬚
■ ■ ■ ■ | ⬚ ⬚ ⬚ ⬚ | ⬚ ⬚ ⬚ ⬚ | ⬚ ⬚ ⬚ ⬚
--------+---------+---------+--------
⬚ ⬚ ⬚ ⬚ | ■ ⬚ ⬚ ⬚ | ⬚ ⬚ ⬚ ⬚ | ⬚ ⬚ ⬚ ⬚
⬚ ⬚ ⬚ ⬚ | ■ ■ ⬚ ⬚ | ⬚ ⬚ ⬚ ⬚ | ⬚ ⬚ ⬚ ⬚
...

压缩为块级表示（4x4 块）:
Block(0,0)=True, Block(0,1)=False, Block(1,0)=False, Block(1,1)=True, ...
```

### 张量形状变化

```
输入:
- cache_position: (q_length,)
- 注意力块: 压缩的 BlockMask 对象

输出:
- BlockMask: 压缩的块级掩码表示
```

### 使用示例

```python
# Flex Attention 掩码
mask = flex_attention_mask(
    batch_size=2,
    cache_position=torch.arange(8),
    kv_length=12,
    mask_function=sliding_window_causal_mask_function(4)
)

# 在 Flex Attention 中使用
# from torch.nn.attention.flex_attention import flex_attention
# out = flex_attention(q, k, v, block_mask=mask)
```

## 实现对比

| 实现方式 | 返回类型 | 优势 | 劣势 |
|---------|---------|------|------|
| SDPA | `Optional[torch.Tensor]` | 高度优化，支持跳过 | 复杂掩码模式支持有限 |
| Eager | `torch.Tensor` | 精确控制，调试友好 | 性能较低，内存占用高 |
| Flash Attention | `Optional[torch.Tensor]` | 极高性能，内存高效 | 仅支持简单掩码模式 |
| Flex Attention | `BlockMask` | 高度灵活，压缩存储 | 需要最新 PyTorch 版本 |
