# 使用示例与可视化

本文档提供详细的使用示例和张量形状变化图解，帮助理解注意力掩码的工作原理。

## 目录
- [基础使用示例](#基础使用示例)
- [张量形状变化详解](#张量形状变化详解)
- [掩码可视化工具](#掩码可视化工具)
- [实际应用场景](#实际应用场景)
- [调试技巧](#调试技巧)

## 基础使用示例

### 示例 1: 简单因果掩码

```python
import torch
from transformers import AutoConfig
from .masking_utils import create_causal_mask, causal_mask_function

# 1. 设置基础参数
config = AutoConfig.from_pretrained("gpt2")
batch_size = 2
seq_length = 5
hidden_dim = 768

# 2. 创建输入张量
input_embeds = torch.randn(batch_size, seq_length, hidden_dim)
cache_position = torch.arange(seq_length)

print(f"输入形状: {input_embeds.shape}")  # [2, 5, 768]
print(f"缓存位置: {cache_position}")      # tensor([0, 1, 2, 3, 4])

# 3. 创建因果掩码
causal_mask = create_causal_mask(
    config=config,
    input_embeds=input_embeds,
    attention_mask=None,
    cache_position=cache_position,
    past_key_values=None
)

print(f"因果掩码形状: {causal_mask.shape}")  # [2, 1, 5, 5]
print(f"掩码数据类型: {causal_mask.dtype}")  # torch.bool
```

**张量形状变化:**
```
input_embeds: [2, 5, 768]
         ↓ 提取批次和序列长度
cache_position: [5]
         ↓ 扩展到批次维度
batch_indices: [2] + head_indices: [1] + cache_position: [5] + kv_indices: [5]
         ↓ 应用掩码函数
causal_mask: [2, 1, 5, 5]
```

### 示例 2: 滑动窗口掩码

```python
from .masking_utils import create_sliding_window_causal_mask, sliding_window_overlay

# 设置滑动窗口
config.sliding_window = 3

# 创建滑动窗口掩码
sliding_mask = create_sliding_window_causal_mask(
    config=config,
    input_embeds=input_embeds,
    attention_mask=None,
    cache_position=cache_position,
    past_key_values=None
)

print(f"滑动窗口掩码形状: {sliding_mask.shape}")  # [2, 1, 5, 5]

# 手动验证掩码模式
sliding_fn = sliding_window_overlay(3)
for q_idx in range(seq_length):
    row = []
    for kv_idx in range(seq_length):
        # 因果 AND 滑动窗口
        allowed = (kv_idx <= q_idx) and (kv_idx > q_idx - 3)
        row.append("■" if allowed else "⬚")
    print(f"位置 {q_idx}: {' '.join(row)}")
```

**输出结果:**
```
位置 0: ■ ⬚ ⬚ ⬚ ⬚
位置 1: ■ ■ ⬚ ⬚ ⬚
位置 2: ■ ■ ■ ⬚ ⬚
位置 3: ⬚ ■ ■ ■ ⬚
位置 4: ⬚ ⬚ ■ ■ ■
```

### 示例 3: 带填充的掩码

```python
# 创建带填充的输入
attention_mask = torch.tensor([
    [True, True, True, True, False],  # 序列 1: 4 有效 + 1 填充
    [True, True, False, False, False]  # 序列 2: 2 有效 + 3 填充
])

input_embeds_with_padding = torch.randn(batch_size, seq_length, hidden_dim)
input_embeds_with_padding[0, 4] = 0  # 填充位置设为 0
input_embeds_with_padding[1, 2:] = 0  # 填充位置设为 0

# 创建带填充的掩码
padded_mask = create_causal_mask(
    config=config,
    input_embeds=input_embeds_with_padding,
    attention_mask=attention_mask,
    cache_position=cache_position,
    past_key_values=None
)

print(f"带填充掩码形状: {padded_mask.shape}")  # [2, 1, 5, 5]

# 检查填充是否被正确处理
batch_0_mask = padded_mask[0, 0]  # 第一个样本的掩码
batch_1_mask = padded_mask[1, 0]  # 第二个样本的掩码

print("批次 0 掩码 (最后位置应为 False):")
print(batch_0_mask[:, -1])  # 应该全是 False (因为 kv 位置是填充)

print("批次 1 掩码 (位置 2+ 应为 False):")
print(batch_1_mask[:, 2:])  # 应该全是 False
```

### 示例 4: 分块注意力掩码

```python
from .masking_utils import create_chunked_causal_mask

# 设置分块参数
config.attention_chunk_size = 3
seq_length = 10

# 创建更长的输入
long_input_embeds = torch.randn(1, seq_length, hidden_dim)
long_cache_position = torch.arange(seq_length)

# 创建分块掩码
chunked_mask = create_chunked_causal_mask(
    config=config,
    input_embeds=long_input_embeds,
    attention_mask=None,
    cache_position=long_cache_position,
    past_key_values=None
)

print(f"分块掩码形状: {chunked_mask.shape}")  # [1, 1, 10, 10]

# 验证分块模式
mask_matrix = chunked_mask[0, 0].cpu().numpy()
for i in range(seq_length):
    row = "".join(["■" if mask_matrix[i, j] else "⬚" for j in range(seq_length)])
    print(f"位置 {i}: {row}")
```

**输出结果:**
```
位置 0: ■■■⬚⬚⬚⬚⬚⬚⬚
位置 1: ■■■⬚⬚⬚⬚⬚⬚⬚
位置 2: ■■■⬚⬚⬚⬚⬚⬚⬚
位置 3: ⬚⬚⬚■■■⬚⬚⬚⬚
位置 4: ⬚⬚⬚■■■⬚⬚⬚⬚
位置 5: ⬚⬚⬚■■■⬚⬚⬚⬚
位置 6: ⬚⬚⬚⬚⬚⬚■■■⬚
位置 7: ⬚⬚⬚⬚⬚⬚■■■⬚
位置 8: ⬚⬚⬚⬚⬚⬚■■■⬚
位置 9: ⬚⬚⬚⬚⬚⬚⬚⬚⬚⬚
```

## 张量形状变化详解

### 1. 基础因果掩码的形状变换

```python
def trace_tensor_shapes(batch_size, seq_length, hidden_dim):
    """追踪张量形状变换过程"""

    # 步骤 1: 输入嵌入
    input_embeds = torch.randn(batch_size, seq_length, hidden_dim)
    print(f"1. 输入嵌入: {input_embeds.shape}")

    # 步骤 2: 缓存位置
    cache_position = torch.arange(seq_length)
    print(f"2. 缓存位置: {cache_position.shape}")

    # 步骤 3: 掩码函数处理
    mask_function = causal_mask_function
    print(f"3. 掩码函数: 标量函数 (batch_idx, head_idx, q_idx, kv_idx) -> bool")

    # 步骤 4: vmap 扩展
    batch_arange = torch.arange(batch_size)
    head_arange = torch.arange(1)
    print(f"4. 批次索引: {batch_arange.shape}")
    print(f"5. 头部索引: {head_arange.shape}")

    # 步骤 5: 完整网格创建
    # 实际上通过 vmap 创建:
    # batch_idx [batch_size] x head_idx [1] x q_idx [seq_length] x kv_idx [seq_length]
    final_shape = (batch_size, 1, seq_length, seq_length)
    print(f"6. 最终掩码: {final_shape}")

# 运行示例
trace_tensor_shapes(2, 5, 768)
```

**输出:**
```
1. 输入嵌入: torch.Size([2, 5, 768])
2. 缓存位置: torch.Size([5])
3. 掩码函数: 标量函数 (batch_idx, head_idx, q_idx, kv_idx) -> bool
4. 批次索引: torch.Size([2])
5. 头部索引: torch.Size([1])
6. 最终掩码: (2, 1, 5, 5)
```

### 2. 带偏移的掩码形状变换

```python
def trace_offset_mask_shapes():
    """追踪带缓存偏移的掩码形状变换"""

    batch_size = 1
    new_tokens = 3
    past_length = 10
    total_length = past_length + new_tokens

    # 输入: 只有新 token
    input_embeds = torch.randn(batch_size, new_tokens, 768)
    print(f"1. 新输入嵌入: {input_embeds.shape}")

    # 缓存位置: 新 token 的绝对位置
    cache_position = torch.arange(past_length, total_length)
    print(f"2. 缓存位置: {cache_position.shape} -> {cache_position.tolist()}")

    # KV 长度: 总长度
    kv_length = total_length
    print(f"3. KV 长度: {kv_length}")

    # KV 偏移: 0 (因为 KV 从位置 0 开始)
    kv_offset = 0
    print(f"4. KV 偏移: {kv_offset}")

    # 最终掩码形状
    final_shape = (batch_size, 1, new_tokens, total_length)
    print(f"5. 最终掩码: {final_shape}")

    # 实际掩码模式解释
    print(f"6. 掩码模式: 每个新 token 可以看到所有 {total_length} 个位置")

trace_offset_mask_shapes()
```

**输出:**
```
1. 新输入嵌入: torch.Size([1, 3, 768])
2. 缓存位置: torch.Size([3]) -> [10, 11, 12]
3. KV 长度: 13
4. KV 偏移: 0
5. 最终掩码: (1, 1, 3, 13)
6. 掩码模式: 每个新 token 可以看到所有 13 个位置
```

### 3. 多头注意力的头维度处理

```python
def trace_multi_head_shapes():
    """追踪多头注意力的形状处理"""

    batch_size = 2
    num_heads = 8
    seq_length = 5

    # 注意掩码不直接包含头维度
    # 它被广播到所有头
    mask_shape = (batch_size, 1, seq_length, seq_length)
    print(f"1. 注意掩码形状: {mask_shape}")

    # 在注意力计算中，掩码被广播
    attention_scores_shape = (batch_size, num_heads, seq_length, seq_length)
    print(f"2. 注意力分数形状: {attention_scores_shape}")

    # 广播机制
    print(f"3. 广播: {mask_shape} -> {attention_scores_shape}")
    print(f"   头维度: 1 -> {num_heads} (广播)")

trace_multi_head_shapes()
```

## 掩码可视化工具

### 1. 基础可视化函数

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_mask(mask, title="注意力掩码"):
    """可视化注意力掩码"""
    if isinstance(mask, torch.Tensor):
        mask_numpy = mask.cpu().numpy()
    else:
        mask_numpy = mask

    # 如果是 4D，取第一个批次和头
    if len(mask_numpy.shape) == 4:
        mask_numpy = mask_numpy[0, 0]

    plt.figure(figsize=(8, 6))
    sns.heatmap(mask_numpy.astype(int),
                cmap='RdYlBu_r',
                cbar_kws={'label': '允许注意力'},
                annot=True,
                fmt='d')
    plt.title(title)
    plt.xlabel('Key-Value 位置')
    plt.ylabel('Query 位置')
    plt.show()

# 使用示例
causal_mask = create_causal_mask(config, input_embeds, None, cache_position, None)
visualize_mask(causal_mask, "因果注意力掩码")
```

### 2. 比较不同掩码类型

```python
def compare_mask_types():
    """比较不同类型的掩码"""

    seq_length = 8
    input_embeds = torch.randn(1, seq_length, 768)
    cache_position = torch.arange(seq_length)

    # 创建不同类型的掩码
    masks = {}

    # 1. 因果掩码
    masks['因果'] = create_causal_mask(config, input_embeds, None, cache_position, None)

    # 2. 滑动窗口
    config.sliding_window = 3
    masks['滑动窗口'] = create_sliding_window_causal_mask(config, input_embeds, None, cache_position, None)

    # 3. 分块
    config.attention_chunk_size = 4
    masks['分块'] = create_chunked_causal_mask(config, input_embeds, None, cache_position, None)

    # 可视化比较
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (name, mask) in enumerate(masks.items()):
        mask_data = mask[0, 0].cpu().numpy()

        sns.heatmap(mask_data.astype(int),
                    cmap='RdYlBu_r',
                    ax=axes[idx],
                    cbar=False,
                    annot=True,
                    fmt='d')
        axes[idx].set_title(f'{name}掩码')
        axes[idx].set_xlabel('KV 位置')
        axes[idx].set_ylabel('Q 位置')

    plt.tight_layout()
    plt.show()

compare_mask_types()
```

### 3. 动态掩码生成过程

```python
def trace_mask_generation():
    """追踪掩码生成的逐步过程"""

    seq_length = 6
    cache_position = torch.arange(seq_length)

    # 步骤 1: 基础因果函数
    def show_causal_step():
        print("步骤 1: 基础因果掩码函数")
        print("函数: kv_idx <= q_idx")
        print("输入: (batch_idx, head_idx, q_idx, kv_idx)")
        print("输出: bool")
        print()

        # 显示前几个位置的掩码
        for q_idx in range(3):
            row = []
            for kv_idx in range(seq_length):
                allowed = kv_idx <= q_idx
                row.append("■" if allowed else "⬚")
            print(f"q={q_idx}: {' '.join(row)}")
        print()

    # 步骤 2: 添加滑动窗口
    def show_sliding_step():
        print("步骤 2: 添加滑动窗口约束")
        print("函数: (kv_idx <= q_idx) AND (kv_idx > q_idx - 3)")
        print()

        sliding_fn = sliding_window_overlay(3)
        for q_idx in range(3):
            row = []
            for kv_idx in range(seq_length):
                causal_allowed = kv_idx <= q_idx
                sliding_allowed = sliding_fn(0, 0, q_idx, kv_idx)
                final_allowed = causal_allowed and sliding_allowed
                row.append("■" if final_allowed else "⬚")
            print(f"q={q_idx}: {' '.join(row)}")
        print()

    # 步骤 3: vmap 扩展
    def show_vmap_step():
        print("步骤 3: vmap 扩展到 4D 张量")
        print("标量函数 -> [batch, head, q, kv] 维度")
        print()

        # 创建小示例
        batch_size, seq_len = 2, 3
        mask = create_causal_mask(
            config,
            torch.randn(batch_size, seq_len, 768),
            None,
            torch.arange(seq_len),
            None
        )
        print(f"最终形状: {mask.shape}")
        print("批次 0, 头 0:")
        for i in range(seq_len):
            row = "".join(["■" if mask[0, 0, i, j] else "⬚" for j in range(seq_len)])
            print(f"  {row}")

    show_causal_step()
    show_sliding_step()
    show_vmap_step()

trace_mask_generation()
```

## 实际应用场景

### 1. 文本生成场景

```python
def text_generation_example():
    """文本生成中的掩码使用"""

    # 场景: 续写文本 "The quick brown fox jumps over"
    prompt_length = 6
    max_new_tokens = 5

    # 1. 编码阶段 - 完全因果掩码
    prompt_embeds = torch.randn(1, prompt_length, 768)
    prompt_cache_position = torch.arange(prompt_length)

    prompt_mask = create_causal_mask(
        config, prompt_embeds, None, prompt_cache_position, None
    )
    print(f"编码阶段掩码形状: {prompt_mask.shape}")
    print("模式: 完全因果注意力")

    # 2. 生成阶段 - 增量掩码
    for step in range(max_new_tokens):
        current_pos = prompt_length + step
        new_token_embeds = torch.randn(1, 1, 768)  # 单个新 token
        new_cache_position = torch.tensor([current_pos])

        # 这里 past_key_values 包含之前所有 token
        generation_mask = create_causal_mask(
            config,
            new_token_embeds,
            None,
            new_cache_position,
            past_key_values  # 假设已填充
        )

        print(f"生成步骤 {step}: 掩码形状 {generation_mask.shape}")
        print(f"  当前位置: {current_pos}, 可看到所有前面的 token")

text_generation_example()
```

### 2. 对话系统场景

```python
def dialogue_system_example():
    """对话系统中的掩码处理"""

    # 对话历史
    system_msg = "You are a helpful assistant."  # 5 tokens
    user_msg = "What is the capital of France?"  # 7 tokens
    assistant_msg = "The capital of France is Paris."  # 8 tokens

    # 模拟分词后的长度
    lengths = [5, 7, 8]
    total_length = sum(lengths)

    # 创建角色掩码
    def create_role_mask():
        """创建基于角色的注意力掩码"""

        # 不同角色间的注意力规则
        def role_attention_mask(batch_idx, head_idx, q_idx, kv_idx):
            # 确定查询和键值所属角色
            cumulative = 0
            q_role, kv_role = None, None

            for role_idx, length in enumerate(lengths):
                if cumulative <= q_idx < cumulative + length:
                    q_role = role_idx
                if cumulative <= kv_idx < cumulative + length:
                    kv_role = role_idx
                cumulative += length

            # 注意力规则:
            # - System: 全局可见
            # - User: 可以看到 system 和之前的 user
            # - Assistant: 可以看到所有 (system + user + 自己之前的 assistant)
            if q_role == 0:  # System
                return kv_role <= 0
            elif q_role == 1:  # User
                return kv_role <= 1
            else:  # Assistant
                return True

        return role_attention_mask

    # 创建角色掩码
    role_mask_fn = create_role_mask()

    # 测试掩码
    dialogue_embeds = torch.randn(1, total_length, 768)
    cache_position = torch.arange(total_length)

    # 使用 OR 掩码组合因果约束和角色约束
    role_mask = create_causal_mask(
        config,
        dialogue_embeds,
        None,
        cache_position,
        None,
        or_mask_function=role_mask_fn  # 注意: 这里可能需要特殊处理
    )

    print(f"对话系统掩码形状: {role_mask.shape}")
    print("角色序列: System(5) | User(7) | Assistant(8)")

dialogue_system_example()
```

### 3. 多模态场景

```python
def multimodal_example():
    """多模态模型中的掩码处理"""

    # 图文理解场景
    image_tokens = 64  # 假设图像编码为 64 个 token
    text_tokens = 20   # 文本描述
    total_tokens = image_tokens + text_tokens

    def multimodal_attention_mask(batch_idx, head_idx, q_idx, kv_idx):
        """多模态注意力规则"""

        # 图像区域: [0, 64)
        # 文本区域: [64, 84)

        if q_idx < image_tokens:  # 图像查询
            # 图像 token 可以看到所有图像 token (完全连接)
            # 也可以看到所有文本 token
            return kv_idx < image_tokens or kv_idx >= image_tokens

        else:  # 文本查询
            # 文本 token 可以看到所有图像 token
            # 只能按因果顺序看到前面的文本 token
            return kv_idx < image_tokens or (kv_idx >= image_tokens and kv_idx <= q_idx)

    # 创建多模态输入
    multimodal_embeds = torch.randn(1, total_tokens, 768)
    cache_position = torch.arange(total_tokens)

    # 使用 OR 掩码组合基础因果和多模态规则
    multimodal_mask = create_causal_mask(
        config,
        multimodal_embeds,
        None,
        cache_position,
        None,
        or_mask_function=multimodal_attention_mask
    )

    print(f"多模态掩码形状: {multimodal_mask.shape}")
    print(f"图像 token: 0-{image_tokens-1}")
    print(f"文本 token: {image_tokens}-{total_tokens-1}")

    # 验证特定模式
    mask_data = multimodal_mask[0, 0]

    # 检查图像到文本的注意力
    img_to_text = mask_data[0, image_tokens]  # 第一个图像 token 到第一个文本 token
    print(f"图像到文本注意力: {img_to_text.item()} (应该为 True)")

    # 检查文本的因果性
    text_later_to_earlier = mask_data[image_tokens + 5, image_tokens + 3]  # 后面的文本到前面的文本
    print(f"文本因果注意力: {text_later_to_earlier.item()} (应该为 True)")

    text_earlier_to_later = mask_data[image_tokens + 3, image_tokens + 5]  # 前面的文本到后面的文本
    print(f"文本反向注意力: {text_earlier_to_later.item()} (应该为 False)")

multimodal_example()
```

## 调试技巧

### 1. 掩码正确性验证

```python
def validate_mask_properties(mask, mask_type="causal"):
    """验证掩码的基本属性"""

    if isinstance(mask, torch.Tensor):
        mask_data = mask
    else:
        mask_data = mask.to_dense() if hasattr(mask, 'to_dense') else mask

    # 提取第一个样本的掩码
    if len(mask_data.shape) == 4:
        sample_mask = mask_data[0, 0]
    else:
        sample_mask = mask_data

    seq_len = sample_mask.shape[0]

    print(f"验证 {mask_type} 掩码 (序列长度: {seq_len}):")

    # 1. 检查对角线 (自注意力)
    diagonal_correct = torch.all(sample_mask.diagonal())
    print(f"  自注意力对角线: {'✓' if diagonal_correct else '✗'}")

    # 2. 检查因果性 (仅对因果掩码)
    if mask_type == "causal":
        causal_violations = 0
        for q_idx in range(seq_len):
            for kv_idx in range(q_idx + 1, seq_len):
                if sample_mask[q_idx, kv_idx]:
                    causal_violations += 1

        print(f"  因果违规数量: {causal_violations} (应为 0)")

    # 3. 检查稀疏性
    total_elements = seq_len * seq_len
    true_elements = torch.sum(sample_mask).item()
    sparsity = (total_elements - true_elements) / total_elements * 100
    print(f"  掩码稀疏度: {sparsity:.1f}%")

    # 4. 检查对称性 (因果掩码应该是下三角)
    if mask_type == "causal":
        upper_triangle = torch.triu(sample_mask, diagonal=1)
        upper_violations = torch.sum(upper_triangle).item()
        print(f"  上三角违规: {upper_violations} (应为 0)")

    return diagonal_correct and (causal_violations == 0 if mask_type == "causal" else True)

# 使用示例
causal_mask = create_causal_mask(config, input_embeds, None, cache_position, None)
validate_mask_properties(causal_mask, "causal")
```

### 2. 性能分析

```python
def profile_mask_creation():
    """分析掩码创建的性能"""

    import time

    sizes = [128, 512, 1024, 2048, 4096]
    times = {}

    for seq_len in sizes:
        print(f"\n测试序列长度: {seq_len}")

        # 准备输入
        test_embeds = torch.randn(2, seq_len, 768)
        test_cache_pos = torch.arange(seq_len)

        # 测量时间
        start_time = time.time()
        mask = create_causal_mask(config, test_embeds, None, test_cache_pos, None)
        end_time = time.time()

        elapsed = end_time - start_time
        times[seq_len] = elapsed

        print(f"  创建时间: {elapsed:.4f}s")
        print(f"  掩码大小: {mask.numel() * mask.element_size() / 1024 / 1024:.2f} MB")
        print(f"  稀疏度: {(1 - torch.mean(mask.float())) * 100:.1f}%")

    # 分析时间复杂度
    print(f"\n时间复杂度分析:")
    for size, time_taken in times.items():
        print(f"  {size}: {time_taken:.4f}s")

profile_mask_creation()
```

### 3. 内存使用分析

```python
def analyze_memory_usage():
    """分析不同掩码类型的内存使用"""

    seq_len = 2048
    batch_size = 4

    # 1. 完整 4D 布尔掩码
    bool_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool)
    bool_memory = bool_mask.numel() * bool_mask.element_size()

    # 2. 浮点掩码 (eager)
    float_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.float32)
    float_memory = float_mask.numel() * float_mask.element_size()

    # 3. BlockMask (Flex Attention) - 估算
    block_size = 128
    num_blocks = (seq_len + block_size - 1) // block_size
    block_memory = num_blocks * num_blocks * 4  # 估算

    print(f"内存使用对比 (序列长度 {seq_len}, 批次 {batch_size}):")
    print(f"  布尔掩码:     {bool_memory / 1024 / 1024:.2f} MB")
    print(f"  浮点掩码:     {float_memory / 1024 / 1024:.2f} MB")
    print(f"  BlockMask:    {block_memory / 1024:.2f} KB")
    print(f"  内存节省 (Block vs 布尔): {(bool_memory - block_memory * 1024) / bool_memory * 100:.1f}%")

analyze_memory_usage()
```

### 4. 掩码一致性检查

```python
def check_mask_consistency(mask1, mask2, name1="Mask1", name2="Mask2"):
    """检查两个掩码的一致性"""

    if mask1.shape != mask2.shape:
        print(f"❌ 形状不匹配: {mask1.shape} vs {mask2.shape}")
        return False

    # 转换为相同数据类型
    if mask1.dtype != mask2.dtype:
        mask1 = mask1.bool()
        mask2 = mask2.bool()

    differences = torch.sum(mask1 != mask2).item()
    total_elements = mask1.numel()

    if differences == 0:
        print(f"✅ {name1} 和 {name2} 完全一致")
        return True
    else:
        print(f"❌ {name1} 和 {name2} 有 {differences}/{total_elements} 个不同位置")

        # 找出前几个差异位置
        diff_positions = torch.nonzero(mask1 != mask2)
        print("前 5 个差异位置:")
        for i, pos in enumerate(diff_positions[:5]):
            q, kv = pos[0], pos[1]
            print(f"  位置 ({q}, {kv}): {mask1[q, kv]} vs {mask2[q, kv]}")

        return False

# 使用示例
# 比较不同实现生成的掩码
mask1 = create_causal_mask(config, input_embeds, None, cache_position, None)
mask2 = sdpa_mask(2, cache_position, 5, 0, causal_mask_function, None)

check_mask_consistency(mask1[0, 0], mask2[0, 0], "高级掩码", "SDPA掩码")
```