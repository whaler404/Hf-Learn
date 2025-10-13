# 生成策略详解

本文档详细分析 Transformers 库中支持的各种文本生成策略，包括其实现原理、适用场景和性能特点。

## 1. 生成策略概览

Transformers 支持以下主要生成策略，通过 `GenerationConfig` 的参数组合来控制：

| 策略 | num_beams | do_sample | 方法名 | 特点 |
|------|-----------|-----------|--------|------|
| 贪心搜索 | 1 | False | `_sample` | 快速，确定性强 |
| 多项式采样 | 1 | True | `_sample` | 随机性强，创造性 |
| Beam Search | >1 | False | `_beam_search` | 质量高，确定性 |
| Beam Sample | >1 | True | `_beam_search` | 质量高，随机性 |
| 辅助生成 | - | - | `_assisted_decoding` | 加速推理 |

## 2. 贪心搜索 (Greedy Search)

### 2.1 原理
贪心搜索在每个时间步选择概率最高的 token 作为输出，是最简单直接的生成策略。

### 2.2 配置参数
```python
generation_config = GenerationConfig(
    do_sample=False,      # 不使用采样
    num_beams=1,         # 单束搜索
    temperature=1.0,     # 不影响贪心搜索
    top_k=50,            # 不影响贪心搜索
    top_p=1.0,           # 不影响贪心搜索
)
```

### 2.3 实现细节

在 `_sample()` 方法中 (`generation/utils.py:2831`):
```python
if do_sample:
    # 采样模式
    probs = nn.functional.softmax(next_token_scores, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
else:
    # 贪心搜索模式
    next_tokens = torch.argmax(next_token_scores, dim=-1)
```

### 2.4 优缺点
**优点**:
- 计算速度快，内存占用低
- 结果确定，可重现
- 适合需要精确输出的场景

**缺点**:
- 容易陷入局部最优
- 缺乏创造性，输出较单调
- 可能产生重复内容

### 2.5 适用场景
- 代码生成
- 结构化文本生成
- 需要高确定性的应用

## 3. 多项式采样 (Multinomial Sampling)

### 3.1 原理
多项式采样根据 token 的概率分布进行随机采样，增加了输出的多样性和创造性。

### 3.2 配置参数
```python
generation_config = GenerationConfig(
    do_sample=True,       # 启用采样
    num_beams=1,         # 单束搜索
    temperature=0.7,     # 控制随机性，<1.0 更保守，>1.0 更随机
    top_k=50,            # 只考虑概率最高的 50 个 token
    top_p=0.9,           # 核采样，累积概率达到 0.9 的最小集合
)
```

### 3.3 关键参数详解

#### temperature (温度)
- **作用**: 调整概率分布的平滑程度
- **范围**: 通常在 0.1-2.0 之间
- **效果**:
  - `temperature < 1.0`: 分布更集中，输出更保守
  - `temperature = 1.0`: 保持原始概率分布
  - `temperature > 1.0`: 分布更平滑，输出更随机

#### top_k (Top-K 采样)
- **作用**: 限制候选 token 数量
- **实现**: 只保留概率最高的 K 个 token
- **效果**: 避免低概率 token 的噪声

#### top_p (Nucleus 采样)
- **作用**: 动态选择候选 token 集合
- **实现**: 选择累积概率达到 P 的最小 token 集合
- **优势**: 适应不同情况，更灵活

### 3.4 实现细节

```python
# 在 logits_processor 中应用
if do_sample:
    # 应用 temperature
    next_token_scores = next_token_scores / temperature

    # 应用 top_k 和 top_p 过滤
    next_token_scores = top_k_filter(next_token_scores, top_k)
    next_token_scores = top_p_filter(next_token_scores, top_p)

    # 多项式采样
    probs = nn.functional.softmax(next_token_scores, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
```

### 3.5 优缺点
**优点**:
- 输出多样性强
- 更具创造性
- 避免重复模式

**缺点**:
- 结果不确定，不可重现
- 可能产生不连贯的内容
- 质量不稳定

### 3.6 适用场景
- 创意写作
- 对话生成
- 开放式任务

## 4. Beam Search

### 4.1 原理
Beam Search 在每个时间步保留多个候选序列（beam），通过累积分数选择最优路径。

### 4.2 配置参数
```python
generation_config = GenerationConfig(
    do_sample=False,         # 不使用采样
    num_beams=4,            # beam 数量
    early_stopping=True,    # 早停策略
    length_penalty=1.0,     # 长度惩罚
    no_repeat_ngram_size=2, # 避免重复
)
```

### 4.3 算法流程

#### 4.3.1 初始化阶段 (`_beam_search:3156-3200`)
```python
# 计算批次大小
batch_size = batch_size_unflattened // num_beams

# 计算需要保留的候选数量
n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
beams_to_keep = max(2, 1 + n_eos_tokens) * num_beams
```

#### 4.3.2 扩展阶段
在每个时间步：
1. 对每个 beam 生成下一个 token 的所有可能
2. 计算每个新序列的累积分数
3. 选择分数最高的 `beams_to_keep` 个候选

#### 4.3.3 选择阶段
```python
# 应用长度惩罚
scores = scores / (sequence_length**length_penalty)

# 选择最佳序列
best_scores, best_indices = torch.topk(scores.flatten(), k=num_return_sequences)
```

### 4.4 关键优化

#### 4.4.1 早停策略
- **True**: 当有 `num_beams` 个完整序列完成时停止
- **False**: 应用启发式方法，当不可能找到更好候选时停止

#### 4.4.2 长度惩罚
```python
# 计算长度惩罚后的分数
final_score = log_prob / (sequence_length**length_penalty)
```
- `length_penalty > 1.0`: 偏向长序列
- `length_penalty < 1.0`: 偏向短序列
- `length_penalty = 1.0`: 无偏好

#### 4.4.3 重复避免
```python
# 避免重复 n-gram
no_repeat_ngram_size = 2  # 避免 2-gram 重复
```

### 4.5 优缺点
**优点**:
- 生成质量高
- 全局优化能力强
- 结果确定

**缺点**:
- 计算复杂度高 (O(num_beams × vocab_size))
- 内存占用大
- 倾向于生成安全但无创意的内容

### 4.6 适用场景
- 翻译任务
- 摘要生成
- 需要高质量输出的应用

## 5. Beam Sample (束采样)

### 5.1 原理
结合 Beam Search 的全局优化能力和采样的多样性，在每个 beam 内部进行采样。

### 5.2 配置参数
```python
generation_config = GenerationConfig(
    do_sample=True,         # 启用采样
    num_beams=4,           # 多 beam
    temperature=0.7,       # 采样温度
    top_p=0.9,            # 核采样
    early_stopping=True,  # 早停
)
```

### 5.3 实现特点
在 `_beam_search()` 方法中，当 `do_sample=True` 时：
```python
if do_sample:
    # 在每个 beam 中进行采样
    probs = nn.functional.softmax(next_token_scores, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=num_beams)
else:
    # 贪心选择
    next_tokens = torch.argmax(next_token_scores, dim=-1)
```

### 5.4 优缺点
**优点**:
- 兼顾质量和多样性
- 比 pure beam search 更有创意
- 比 pure sampling 更连贯

**缺点**:
- 计算复杂度最高
- 参数调优困难
- 结果不可重现

### 5.5 适用场景
- 创意写作需要高质量时
- 需要平衡创造性和连贯性的任务

## 6. 辅助生成 (Assisted Generation)

### 6.1 原理
使用小型的辅助模型（assistant model）快速生成候选 token，然后由大型目标模型验证，显著加速推理过程。

### 6.2 配置参数
```python
generation_config = GenerationConfig(
    num_assistant_tokens=20,           # 每次迭代的候选 token 数
    assistant_confidence_threshold=0.4, # 辅助模型置信度阈值
    num_assistant_tokens_schedule="constant", # 调度策略
    use_cache=True,                   # 必须启用缓存
)
```

### 6.3 实现流程

#### 6.3.1 候选生成器选择 (`_assisted_decoding:3512-3521`)
```python
candidate_generator = self._get_candidate_generator(
    generation_config=generation_config,
    input_ids=input_ids,
    inputs_tensor=inputs_tensor,
    assistant_model=assistant_model,
    logits_processor=logits_processor,
    target_tokenizer=tokenizer,
    assistant_tokenizer=assistant_tokenizer,
    model_kwargs=model_kwargs,
)
```

#### 6.3.2 辅助模型推理
1. 辅助模型快速生成多个候选 token
2. 目标模型批量验证候选 token
3. 接受正确的 token，拒绝错误的 token
4. 重复直到遇到错误的 token

#### 6.3.3 调度策略
- **"constant"**: 保持固定的候选 token 数量
- **"heuristic"**: 动态调整，成功时增加，失败时减少
- **"heuristic_transient"**: 同上，但每次调用重置

### 6.4 关键约束
```python
# 必须使用动态缓存
if not model_kwargs["use_cache"]:
    raise ValueError("assisted generate requires `use_cache=True`")

# 不支持静态缓存
if generation_config.cache_implementation in ["static", "hybrid", "sliding_window"]:
    raise ValueError("assisted generate is not supported with Static cache classes")
```

### 6.5 优缺点
**优点**:
- 显著加速推理（2-5倍）
- 保持生成质量
- 透明优化，用户无感知

**缺点**:
- 需要额外的辅助模型
- 内存占用增加
- 辅助模型质量影响效果

### 6.6 适用场景
- 大规模推理服务
- 实时应用
- 对延迟敏感的场景

## 7. 核心生成方法实现详解

### 7.1 `_sample()` 方法详细分析

`_sample()` 方法 (`generation/utils.py:2686`) 是最基础的生成方法，同时支持贪心搜索和多项式采样。

 主生成循环详解

```python
while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
    # 1. 准备模型输入
    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

    # 2. 模型前向传播
    if is_prefill:
        outputs = self(**model_inputs, return_dict=True)
        is_prefill = False
    else:
        outputs = model_forward(**model_inputs, return_dict=True)

    # 3. 同步 GPU 处理
    model_kwargs = self._update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
    )
    if synced_gpus and this_peer_finished:
        continue

    # 4. 获取下一个 token 的 logits
    next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

    # 5. 应用 logits 处理器
    next_token_scores = logits_processor(input_ids, next_token_logits)

    # 6. 存储中间结果（如果需要）
    if return_dict_in_generate:
        if output_scores:
            scores += (next_token_scores,)
        if output_logits:
            raw_logits += (next_token_logits,)
        if output_attentions:
            decoder_attentions += (
                (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
            )
            if self.config.is_encoder_decoder:
                cross_attentions += (outputs.cross_attentions,)
        if output_hidden_states:
            decoder_hidden_states += (
                (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)
            )

    # 7. Token 选择（核心逻辑）
    if do_sample:
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        next_tokens = torch.argmax(next_token_scores, dim=-1)

    # 8. 处理已完成的序列
    if has_eos_stopping_criteria:
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

    # 9. 更新序列和状态
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    if streamer is not None:
        streamer.put(next_tokens.cpu())

    unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
    this_peer_finished = unfinished_sequences.max() == 0
    cur_len += 1

    # 10. 清理内存
    del outputs
```

### 7.2 `_beam_search()` 方法详细分析

`_beam_search()` 方法 (`generation/utils.py:3111`) 实现了束搜索算法，支持贪心和采样两种模式。

#### 7.2.1 Beam 缓冲区设置

```python
# 计算需要保留的候选数量（考虑 EOS token）
n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
beams_to_keep = max(2, 1 + n_eos_tokens) * num_beams

# 创建 beam 掩码
top_num_beam_mask = torch.cat(
    (torch.ones((num_beams), dtype=torch.bool), torch.zeros((beams_to_keep - num_beams), dtype=torch.bool)),
    dim=0,
).to(input_ids.device)

# 初始化缓存位置
model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
```

#### 7.2.2 初始化输出存储

```python
# 初始化输出张量
all_scores = () if (return_dict_in_generate and output_scores) else None
raw_logits = () if (return_dict_in_generate and output_logits) else None
beam_indices = () if (return_dict_in_generate and output_logits) else None

# 初始化 beam scores
beam_scores = torch.zeros((batch_size,), dtype=torch.float, device=input_ids.device)
beam_scores = beam_scores.unsqueeze(-1).expand(batch_size, num_beams)

# 初始化 beam 令牌
beam_tokens = torch.zeros((batch_size, num_beams), dtype=torch.long, device=input_ids.device)
beam_indices = torch.zeros((batch_size, num_beams), dtype=torch.long, device=input_ids.device)
```

#### 7.2.3 主搜索循环

```python
while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
    # 1. 准备模型输入
    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

    # 2. 模型前向传播
    outputs = self(**model_inputs, return_dict=True)

    # 3. 更新模型参数
    model_kwargs = self._update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
    )

    # 4. 获取下一个 token 的 logits
    next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

    # 5. 应用 logits 处理器
    next_token_scores = logits_processor(input_ids, next_token_logits)

    # 6. 存储中间结果
    if return_dict_in_generate:
        if output_scores:
            all_scores += (next_token_scores,)
        if output_logits:
            raw_logits += (next_token_logits,)

    # 7. 计算下一个 token 分数
    vocab_size = next_token_scores.shape[-1]
    next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)

    # 8. 重塑分数张量用于选择
    next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

    # 9. 选择 top-k 候选
    next_token_scores, next_tokens = torch.topk(
        next_token_scores, beams_to_keep, dim=1
    )

    # 10. 计算 beam 索引
    next_beam_indices = next_tokens // vocab_size
    next_tokens = next_tokens % vocab_size

    # 11. 更新 beam 分数
    beam_scores = next_token_scores[:, :num_beams]

    # 12. 重新排序输入序列
    input_ids = torch.cat([
        input_ids[batch_indices], next_tokens[:, :num_beams].unsqueeze(-1)
    ], dim=-1)

    # 13. 更新缓存
    if "past_key_values" in model_kwargs:
        past_key_values = model_kwargs["past_key_values"]
        past_key_values.reorder_cache(next_beam_indices[:, :num_beams])
        model_kwargs["past_key_values"] = past_key_values

    # 14. 检查停止条件
    if stopping_criteria(input_ids, None):
        break

    # 15. 清理内存
    del outputs
```

#### 7.2.5 输出选择和格式化

```python
# 应用长度惩罚
if length_penalty != 1.0:
    beam_scores = beam_scores / (input_ids.shape[1] ** length_penalty)

# 选择最佳序列
best_scores, best_indices = torch.topk(beam_scores.flatten(), k=num_return_sequences)

# 计算批次和 beam 索引
best_batch_indices = best_indices // num_beams
best_beam_indices = best_indices % num_beams

# 重新排序输出序列
output_sequences = input_ids[best_batch_indices, best_beam_indices]

# 返回结果
if return_dict_in_generate:
    return GenerateBeamDecoderOnlyOutput(
        sequences=output_sequences,
        sequence_scores=best_scores,
        scores=all_scores,
        logits=raw_logits,
        beam_indices=beam_indices,
    )
else:
    return output_sequences
```

### 7.3 `_assisted_decoding()` 方法详细分析

`_assisted_decoding()` 方法 (`generation/utils.py:3445`) 实现了辅助生成算法。

辅助生成主循环

```python
# 初始化序列状态
batch_size, cur_len = input_ids.shape[:2]
this_peer_finished = False
unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
scores = () if (return_dict_in_generate and output_scores) else None
raw_logits = () if (return_dict_in_generate and output_logits) else None

# 初始化模型前向传播
model_forward = self.__call__
compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
if compile_forward:
    model_forward = self.get_compiled_call(generation_config.compile_config)

# 主生成循环
while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
    # 1. 使用候选生成器生成候选 token
    candidate_outputs = candidate_generator.generate_candidates(
        input_ids=input_ids,
        num_return_sequences=generation_config.num_assistant_tokens,
        **model_kwargs
    )

    # 2. 准备模型输入
    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

    # 3. 目标模型验证候选
    outputs = model_forward(**model_inputs, return_dict=True)

    # 4. 更新模型参数
    model_kwargs = self._update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
    )

    # 5. 获取目标模型 logits
    next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
    next_token_scores = logits_processor(input_ids, next_token_logits)

    # 6. 验证候选 token
    accepted_tokens = []
    accepted_logits = []
    accepted_scores = []

    for i, candidate_token in enumerate(candidate_outputs.tokens):
        # 检查候选 token 是否在目标模型的高概率位置
        top_k_indices = torch.topk(next_token_scores[i], k=5).indices
        if candidate_token in top_k_indices:
            accepted_tokens.append(candidate_token)
            accepted_logits.append(next_token_logits[i])
            accepted_scores.append(next_token_scores[i, candidate_token])
        else:
            break  # 遇到错误 token，停止接受

    # 7. 更新序列
    if accepted_tokens:
        # 接受的 token
        new_tokens = torch.tensor(accepted_tokens, device=input_ids.device).unsqueeze(0)
        input_ids = torch.cat([input_ids, new_tokens], dim=-1)

        # 更新分数和日志
        if return_dict_in_generate and output_scores:
            scores += (torch.stack(accepted_scores),)
        if return_dict_in_generate and output_logits:
            raw_logits += (torch.stack(accepted_logits),)
    else:
        # 没有候选被接受，使用标准采样
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(0)], dim=-1)

    # 8. 流处理器更新
    if streamer is not None:
        streamer.put(input_ids[:, -1:].cpu())

    # 9. 检查停止条件
    unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
    this_peer_finished = unfinished_sequences.max() == 0

    # 10. 清理内存
    del outputs

    # 11. 更新候选生成器状态
    candidate_generator.update_state(accepted_tokens)
```


### 7.4 方法性能对比

| 方法 | 时间复杂度 | 空间复杂度 | 并行度 | 质量保证 |
|------|------------|------------|--------|----------|
| `_sample()` | O(n × vocab_size) | O(n) | 低 | 局部最优 |
| `_beam_search()` | O(n × beam_size × vocab_size) | O(n × beam_size) | 中 | 全局最优 |
| `_assisted_decoding()` | O(n × (1 + assistant_speedup)) | O(n) | 高 | 近似最优 |
