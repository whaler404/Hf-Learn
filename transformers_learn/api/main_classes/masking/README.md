# 注意力掩码系统文档

本文档详细介绍 Transformers 库中注意力掩码的设计、实现和使用方法。注意力掩码是 Transformer 模型的核心组件，控制模型在生成过程中的信息访问模式。

## 📚 文档结构

### 🏗️ [基础掩码函数](./basic_mask_functions.md)
- **因果掩码函数** (`causal_mask_function`): 基础的自回归掩码
- **滑动窗口叠加** (`sliding_window_overlay`): 局部注意力模式
- **分块叠加** (`chunked_overlay`): 分块注意力模式
- **填充掩码函数** (`padding_mask_function`): 处理序列填充
- **打包序列掩码函数** (`packed_sequence_mask_function`): 处理打包序列

### 🔧 [掩码组合函数](./mask_composition.md)
- **AND 掩码组合** (`and_masks`): 多个掩码的交集
- **OR 掩码组合** (`or_masks`): 多个掩码的并集
- **掩码偏移函数** (`add_offsets_to_mask_function`): 处理缓存偏移
- 实际应用示例和性能优化技巧

### ⚡ [注意力实现掩码](./attention_implementations.md)
- **SDPA 掩码** (`sdpa_mask`): PyTorch 标准注意力实现
- **Eager 掩码** (`eager_mask`): 显式注意力计算掩码
- **Flash Attention 掩码** (`flash_attention_mask`): 高性能 Flash Attention
- **Flex Attention 掩码** (`flex_attention_mask`): 灵活的块级掩码
- 掩码接口选择和性能对比

### 🎯 [高级掩码创建](./high_level_mask_creation.md)
- **因果掩码创建** (`create_causal_mask`): 完整因果掩码创建
- **滑动窗口因果掩码创建** (`create_sliding_window_causal_mask`): 滑动窗口实现
- **分块因果掩码创建** (`create_chunked_causal_mask`): 分块注意力实现
- **生成掩码创建** (`create_masks_for_generate`): 生成场景的掩码处理
- 预处理器函数和混合层支持

### 📊 [使用示例与可视化](./examples_and_visualizations.md)
- 基础使用示例和代码演示
- 详细的张量形状变化图解
- 掩码可视化工具和比较方法
- 实际应用场景（文本生成、对话系统、多模态）
- 调试技巧和性能分析

## 🚀 快速开始

### 基础因果掩码

```python
import torch
from transformers import AutoConfig
from transformers.utils.masking_utils import create_causal_mask

# 设置基础参数
config = AutoConfig.from_pretrained("gpt2")
batch_size = 2
seq_length = 5
hidden_dim = 768

# 创建输入
input_embeds = torch.randn(batch_size, seq_length, hidden_dim)
cache_position = torch.arange(seq_length)

# 创建因果掩码
causal_mask = create_causal_mask(
    config=config,
    input_embeds=input_embeds,
    attention_mask=None,
    cache_position=cache_position,
    past_key_values=None
)

print(f"掩码形状: {causal_mask.shape}")  # [2, 1, 5, 5]
```

### 滑动窗口掩码

```python
# 配置滑动窗口
config.sliding_window = 3

# 创建滑动窗口掩码
sliding_mask = create_sliding_window_causal_mask(
    config=config,
    input_embeds=input_embeds,
    attention_mask=None,
    cache_position=cache_position,
    past_key_values=None
)
```

### 分块注意力掩码

```python
# 配置分块大小
config.attention_chunk_size = 4

# 创建分块掩码
chunked_mask = create_chunked_causal_mask(
    config=config,
    input_embeds=input_embeds,
    attention_mask=None,
    cache_position=cache_position,
    past_key_values=None
)
```

## 🎨 掩码模式可视化

### 因果掩码 (Causal Mask)
```
q\kv 0 1 2 3 4
  0  ■ ⬚ ⬚ ⬚ ⬚
  1  ■ ■ ⬚ ⬚ ⬚
  2  ■ ■ ■ ⬚ ⬚
  3  ■ ■ ■ ■ ⬚
  4  ■ ■ ■ ■ ■
```

### 滑动窗口 (Sliding Window, window=3)
```
q\kv 0 1 2 3 4
  0  ■ ⬚ ⬚ ⬚ ⬚
  1  ■ ■ ⬚ ⬚ ⬚
  2  ■ ■ ■ ⬚ ⬚
  3  ⬚ ■ ■ ■ ⬚
  4  ⬚ ⬚ ■ ■ ■
```

### 分块注意力 (Chunked Attention, chunk_size=3)
```
q\kv 0 1 2 3 4 5
  0  ■ ■ ■ ⬚ ⬚ ⬚
  1  ■ ■ ■ ⬚ ⬚ ⬚
  2  ■ ■ ■ ⬚ ⬚ ⬚
  3  ⬚ ⬚ ⬚ ■ ■ ■
  4  ⬚ ⬚ ⬚ ■ ■ ■
  5  ⬚ ⬚ ⬚ ■ ■ ■
```

## 🔍 核心概念

### 掩码函数签名
所有基础掩码函数都遵循统一的接口：
```python
def mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool
```

### 张量形状变换
```
输入: 标量索引 (batch_idx, head_idx, q_idx, kv_idx)
     ↓ vmap 扩展
输出: 4D 掩码张量 (batch_size, num_heads, seq_len, seq_len)
```

### 注意力后端选择
```python
# 自动根据配置选择最优后端
mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

# 可用后端:
# - "sdpa": PyTorch 标准注意力 (推荐)
# - "eager": 显式计算 (调试用)
# - "flash_attention_2": 高性能 Flash Attention
# - "flex_attention": 灵活块级注意力
```

## 🎯 应用场景

### 文本生成
- **编码阶段**: 完全因果掩码
- **生成阶段**: 增量掩码 + 缓存优化

### 长序列处理
- **滑动窗口**: Mistral 风格的局部注意力
- **分块注意力**: Llama4 风格的块状注意力

### 多模态模型
- **图文理解**: 图像全局 + 文本因果
- **对话系统**: 角色基础的注意力规则

### 特殊应用
- **打包序列**: 多序列并行处理
- **混合架构**: 不同层使用不同注意力模式

## ⚡ 性能优化

### 掩码跳过优化
```python
# 允许在简单场景下跳过掩码创建
mask = sdpa_mask(
    batch_size=batch_size,
    cache_position=cache_position,
    kv_length=kv_length,
    allow_is_causal_skip=True  # 关键优化
)
```

### 内存优化
- **BlockMask**: Flex Attention 的压缩表示
- **稀疏存储**: 避免稠密张量存储
- **设备对齐**: 减少 GPU-CPU 数据传输

### 编译优化
- **Torch 编译**: 静态掩码预编译
- **JIT 追踪**: 动态掩码的即时编译

## 🛠️ 调试与验证

### 掩码正确性检查
```python
from transformers.utils.masking_utils import validate_mask_properties

# 验证因果掩码属性
is_valid = validate_mask_properties(causal_mask, "causal")
```

### 可视化工具
```python
from transformers.utils.masking_utils import tensor_to_mask_visual

# 可视化掩码模式
visualization = tensor_to_mask_visual(causal_mask[0, 0])
print(visualization)
```

### 性能分析
```python
# 分析不同序列长度下的性能
for seq_len in [512, 1024, 2048, 4096]:
    start_time = time.time()
    mask = create_causal_mask(config, test_embeds, None, cache_pos, None)
    print(f"序列长度 {seq_len}: {time.time() - start_time:.4f}s")
```

## 📖 技术细节

### 版本兼容性
- **PyTorch >= 2.6**: 完整功能支持
- **PyTorch >= 2.5**: 基础功能支持
- **PyTorch < 2.5**: 限制功能支持

### 设备支持
- **CUDA**: 完全支持，性能优化
- **CPU**: 完全支持
- **XPU**: 完全支持，特殊优化
- **MPS**: 基础支持

### 数据类型
- **bool**: 标准掩码格式
- **float32**: Eager 注意力格式
- **float16**: 半精度优化
- **bfloat16**: 混合精度训练

## 🔮 高级特性

### 混合缓存架构
```python
# 支持不同层使用不同缓存策略
if hasattr(past_key_values, "is_sliding"):
    # 混合缓存: 部分层使用滑动窗口
    layer_idx = past_key_values.is_sliding.index(True)
```

### 自定义掩码函数
```python
# 定义自定义注意力模式
def custom_mask_function(batch_idx, head_idx, q_idx, kv_idx):
    # 实现自定义逻辑
    return custom_condition(q_idx, kv_idx)

# 组合到高级掩码中
custom_mask = create_causal_mask(
    config=config,
    input_embeds=input_embeds,
    attention_mask=None,
    cache_position=cache_position,
    past_key_values=None,
    or_mask_function=custom_mask_function
)
```

### 动态掩码生成
```python
# 根据输入动态生成掩码模式
def adaptive_masking(input_length, complexity_threshold):
    if input_length > complexity_threshold:
        return "sliding_attention"
    elif input_length > complexity_threshold // 2:
        return "chunked_attention"
    else:
        return "full_attention"
```

## 🤝 贡献指南

### 添加新的掩码类型
1. 在 `basic_mask_functions.py` 中定义基础函数
2. 在 `high_level_mask_creation.py` 中添加创建函数
3. 在 `LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING` 中注册
4. 添加相应的测试和文档

### 性能优化建议
1. 使用 `torch.vmap` 进行向量化
2. 避免不必要的张量创建
3. 利用设备特定的优化
4. 考虑内存访问模式

## 📚 参考资料

- [PyTorch Attention 文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Flash Attention 原论文](https://arxiv.org/abs/2205.14135)
- [Flex Introduction](https://pytorch.org/blog/flexattention/)
- [Transformers 文档](https://huggingface.co/docs/transformers/)

## 📄 许可证

本文档遵循与 Transformers 库相同的 Apache 2.0 许可证。

---

**注意**: 这是一个活文档，会随着库的发展不断更新。如有问题或建议，请提交 Issue 或 PR。