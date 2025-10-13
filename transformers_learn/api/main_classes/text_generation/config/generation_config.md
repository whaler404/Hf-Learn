# GenerationConfig

`GenerationConfig` 类存储了用于文本生成任务的配置参数。它控制着模型生成文本的各种策略和约束。

## 类定义

```python
class GenerationConfig(PushToHubMixin):
    """存储文本生成配置的类"""
```

## 参数分类

GenerationConfig 的参数可以分为以下几个主要类别：

### 1. 输出长度控制参数

这些参数控制生成文本的长度限制。

#### `max_length`
- **类型**: `int`
- **默认值**: `20`
- **含义**: 生成文本的最大长度（包括输入提示的长度）
- **说明**: 对应 `输入提示长度 + max_new_tokens`。如果同时设置了 `max_new_tokens`，该参数会被覆盖。

#### `max_new_tokens`
- **类型**: `int` (可选)
- **默认值**: `None`
- **含义**: 要生成的最大 token 数量，忽略提示中的 token 数量
- **说明**: 这是控制生成长度的首选参数

#### `min_length`
- **类型**: `int`
- **默认值**: `0`
- **含义**: 生成序列的最小长度（包括输入提示）
- **说明**: 对应 `输入提示长度 + min_new_tokens`。如果同时设置了 `min_new_tokens`，该参数会被覆盖。

#### `min_new_tokens`
- **类型**: `int` (可选)
- **默认值**: `None`
- **含义**: 要生成的最小 token 数量，忽略提示中的 token 数量

#### `early_stopping`
- **类型**: `bool` 或 `str`
- **默认值**: `False`
- **含义**: 控制基于 beam 方法的停止条件
- **可选值**:
  - `True`: 一旦有 `num_beams` 个完整候选序列就停止生成
  - `False`: 应用启发式方法，当不太可能找到更好的候选序列时停止
  - `"never"`: 仅当无法找到更好的候选序列时停止（标准 beam search 算法）

#### `max_time`
- **类型**: `float` (可选)
- **默认值**: `None`
- **含义**: 允许计算运行的最大时间（秒）
- **说明**: 即使超时，生成仍会完成当前轮次的计算

#### `stop_strings`
- **类型**: `str` 或 `list[str]` (可选)
- **默认值**: `None`
- **含义**: 如果模型输出这些字符串，应该终止生成

### 2. 生成策略参数

这些参数控制使用哪种文本生成策略。

#### `do_sample`
- **类型**: `bool`
- **默认值**: `False`
- **含义**: 是否使用采样；否则使用贪心解码
- **说明**: 这是控制解码策略的核心参数

#### `num_beams`
- **类型**: `int`
- **默认值**: `1`
- **含义**: beam search 的 beam 数量
- **说明**: 1 表示不使用 beam search

### 3. 缓存相关参数

这些参数控制 KV 缓存的使用以加速解码。

#### `use_cache`
- **类型**: `bool`
- **默认值**: `True`
- **含义**: 模型是否应该使用过去的键值注意力（如果模型支持）
- **说明**: 用于加速解码

#### `cache_implementation`
- **类型**: `str` (可选)
- **默认值**: `None`
- **含义**: 在 `generate` 中实例化的缓存类名称
- **可选值**:
  - `"dynamic"`: [`DynamicCache`]
  - `"static"`: [`StaticCache`]
  - `"offloaded"`: [`DynamicCache(offloaded=True)`]
  - `"offloaded_static"`: [`StaticCache(offloaded=True)`]
  - `"quantized"`: [`QuantizedCache`]
- **说明**: 如果未指定，将使用模型的默认缓存（通常是 `DynamicCache`）

#### `cache_config`
- **类型**: `dict` (可选)
- **默认值**: `None`
- **含义**: 传递给键值缓存类的参数

#### `return_legacy_cache`
- **类型**: `bool` (可选)
- **默认值**: `True`
- **含义**: 当使用 `DynamicCache` 时，是否返回传统格式或新格式的缓存

#### `prefill_chunk_size`
- **类型**: `int` (可选)
- **默认值**: `None`
- **含义**: 预填充的块大小

### 4. Logits 处理参数

这些参数控制模型输出 logits 的处理方式。

#### `temperature`
- **类型**: `float`
- **默认值**: `1.0`
- **含义**: 用于调节下一个 token 概率的值
- **说明**:
  - 值越高（>1.0）使生成更加随机和创造性
  - 值越低（<1.0）使生成更加确定和保守
  - 设置为 1.0 时保持原始概率分布

#### `top_k`
- **类型**: `int`
- **默认值**: `50`
- **含义**: 为 top-k 过滤保留的最高概率词汇 token 的数量
- **说明**: 仅考虑概率最高的 k 个 token

#### `top_p`
- **类型**: `float`
- **默认值**: `1.0`
- **含义**: 如果设置为小于 1 的浮点数，只保留概率加起来达到或超过 `top_p` 的最小最可能 token 集合
- **说明**: 也称为 nucleus sampling
  - 值越低（如 0.9）使选择更加集中
  - 值为 1.0 时相当于不进行 top-p 过滤

#### `min_p`
- **类型**: `float` (可选)
- **默认值**: `None`
- **含义**: 最小 token 概率，将乘以最可能 token 的概率
- **取值范围**: 0 到 1 之间
- **说明**: 典型值在 0.01-0.2 范围内，与设置 `top_p` 在 0.99-0.8 范围内具有可比性

#### `typical_p`
- **类型**: `float`
- **默认值**: `1.0`
- **含义**: 局部典型性衡量预测目标 token 的条件概率与预测随机 token 的预期条件概率的相似程度
- **说明**: 如果设置为小于 1 的浮点数，保留概率加起来达到或超过 `typical_p` 的最小最局部典型 token 集合

#### `epsilon_cutoff`
- **类型**: `float`
- **默认值**: `0.0`
- **含义**: 如果设置为严格介于 0 和 1 之间的浮点数，只有条件概率大于 `epsilon_cutoff` 的 token 才会被采样
- **说明**: 建议值范围为 3e-4 到 9e-4，取决于模型大小

#### `eta_cutoff`
- **类型**: `float`
- **默认值**: `0.0`
- **含义**: Eta 采样是局部典型采样和 epsilon 采样的混合
- **说明**: 如果设置为严格介于 0 和 1 之间的浮点数，token 只有大于 `eta_cutoff` 或 `sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits)))` 才会被考虑

#### `repetition_penalty`
- **类型**: `float`
- **默认值**: `1.0`
- **含义**: 重复惩罚参数
- **说明**:
  - 1.0 表示无惩罚
  - 值 >1.0 抑制重复
  - 值 <1.0 鼓励重复

#### `encoder_repetition_penalty`
- **类型**: `float`
- **默认值**: `1.0`
- **含义**: 编码器重复惩罚参数
- **说明**: 对不在原始输入中的序列进行指数惩罚。1.0 表示无惩罚。

#### `length_penalty`
- **类型**: `float`
- **默认值**: `1.0`
- **含义**: 与基于 beam 的生成一起使用的长度指数惩罚
- **说明**: 应用于序列长度的指数，用于除以序列分数
  - `length_penalty` > 0.0 促进更长的序列
  - `length_penalty` < 0.0 鼓励更短的序列

#### `no_repeat_ngram_size`
- **类型**: `int`
- **默认值**: `0`
- **含义**: 如果设置为 > 0 的整数，该大小的所有 ngram 只能出现一次
- **说明**: 用于避免重复特定长度的 n-gram

#### `bad_words_ids`
- **类型**: `list[list[int]]` (可选)
- **默认值**: `None`
- **含义**: 不允许生成的 token id 列表的列表
- **说明**: 检查 [`~generation.NoBadWordsLogitsProcessor`] 了解更多文档和示例

#### `renormalize_logits`
- **类型**: `bool`
- **默认值**: `False`
- **含义**: 在应用所有 logits 处理器（包括自定义的）之后是否重新归一化 logits
- **说明**: 强烈建议将此标志设置为 `True`，因为搜索算法假设分数 logits 是归一化的，但一些 logits 处理器会破坏归一化

#### `forced_bos_token_id`
- **类型**: `int` (可选)
- **默认值**: `model.config.forced_bos_token_id`
- **含义**: 强制作为 `decoder_start_token_id` 之后第一个生成 token 的 id
- **说明**: 对于像 [mBART](../model_doc/mbart) 这样的多语言模型很有用，其中第一个生成的 token 需要是目标语言 token

#### `forced_eos_token_id`
- **类型**: `int` 或 `list[int]` (可选)
- **默认值**: `model.config.forced_eos_token_id`
- **含义**: 当达到 `max_length` 时强制作为最后生成 token 的 id
- **说明**: 可选择使用列表设置多个结束序列 token

#### `remove_invalid_values`
- **类型**: `bool`
- **默认值**: `model.config.remove_invalid_values`
- **含义**: 是否移除模型的可能的 *nan* 和 *inf* 输出以防止生成方法崩溃
- **说明**: 注意使用 `remove_invalid_values` 可能会减慢生成速度

#### `exponential_decay_length_penalty`
- **类型**: `tuple(int, float)` (可选)
- **默认值**: `None`
- **含义**: 在生成一定数量的 token 后添加指数增长的长度惩罚
- **说明**: 元组应包含：`(start_index, decay_factor)`，其中 `start_index` 表示惩罚开始的位置，`decay_factor` 表示指数衰减的因子

#### `suppress_tokens`
- **类型**: `list[int]` (可选)
- **默认值**: `None`
- **含义**: 将在生成时被抑制的 token 列表
- **说明**: `SuppressTokens` logit 处理器会将它们的 log probs 设置为 `-inf`，因此它们不会被采样

#### `begin_suppress_tokens`
- **类型**: `list[int]` (可选)
- **默认值**: `None`
- **含义**: 将在生成开始时被抑制的 token 列表
- **说明**: `SuppressBeginTokens` logit 处理器会将它们的 log probs 设置为 `-inf`，因此它们不会被采样

#### `sequence_bias`
- **类型**: `dict[tuple[int], float]` (可选)
- **默认值**: `None`
- **含义**: 将 token 序列映射到其偏置项的字典
- **说明**:
  - 正偏置增加序列被选中的几率
  - 负偏置起到相反作用
  - 检查 [`~generation.SequenceBiasLogitsProcessor`] 了解更多文档和示例

#### `token_healing`
- **类型**: `bool`
- **默认值**: `False`
- **含义**: 通过用适当的扩展替换提示的尾部 token 来修复它们
- **说明**: 这提高了受贪婪标记化偏差影响的提示的完成质量

#### `guidance_scale`
- **类型**: `float` (可选)
- **默认值**: `None`
- **含义**: 无分类器指导（CFG）的指导比例
- **说明**: 通过设置 `guidance_scale > 1` 启用 CFG。更高的指导比例鼓励模型生成与输入提示更紧密相关的样本，通常以质量较差为代价。

#### `watermarking_config`
- **类型**: `BaseWatermarkingConfig` 或 `dict` (可选)
- **默认值**: `None`
- **含义**: 用于通过向随机选择的"绿色"token 集合添加小偏置来对模型输出进行水印的参数
- **说明**: 查看 [`SynthIDTextWatermarkingConfig`] 和 [`WatermarkingConfig`] 的文档了解更多详情。如果作为 `Dict` 传递，将在内部转换为 `WatermarkingConfig`。

### 5. 输出控制参数

这些参数定义 `generate` 的输出变量。

#### `num_return_sequences`
- **类型**: `int`
- **默认值**: `1`
- **含义**: 为批次中每个元素独立计算的返回序列数量

#### `output_attentions`
- **类型**: `bool`
- **默认值**: `False`
- **含义**: 是否返回所有注意力层的注意力张量
- **说明**: 查看返回张量下的 `attentions` 了解更多详情

#### `output_hidden_states`
- **类型**: `bool`
- **默认值**: `False`
- **含义**: 是否返回所有层的隐藏状态
- **说明**: 查看返回张量下的 `hidden_states` 了解更多详情

#### `output_scores`
- **类型**: `bool`
- **默认值**: `False`
- **含义**: 是否返回预测分数
- **说明**: 查看返回张量下的 `scores` 了解更多详情

#### `output_logits`
- **类型**: `bool` (可选)
- **默认值**: `None`
- **含义**: 是否返回未处理的预测 logit 分数
- **说明**: 查看返回张量下的 `logits` 了解更多详情

#### `return_dict_in_generate`
- **类型**: `bool`
- **默认值**: `False`
- **含义**: 是否返回 [`~utils.ModelOutput`]，而不是仅返回生成的序列
- **说明**: 如果要返回生成缓存（当 `use_cache` 为 `True` 时）或可选输出（参见以 `output_` 开头的标志），必须将此标志设置为 `True`

### 6. 特殊标记参数

这些参数定义可以在生成时使用的特殊标记。

#### `pad_token_id`
- **类型**: `int` (可选)
- **默认值**: `None`
- **含义**: *填充* token 的 id

#### `bos_token_id`
- **类型**: `int` (可选)
- **默认值**: `None`
- **含义**: *序列开始* token 的 id

#### `eos_token_id`
- **类型**: `Union[int, list[int]]` (可选)
- **默认值**: `None`
- **含义**: *序列结束* token 的 id
- **说明**: 可选择使用列表设置多个结束序列 token

### 7. 编码器-解码器模型专用参数

这些参数专门用于编码器-解码器模型。

#### `encoder_no_repeat_ngram_size`
- **类型**: `int`
- **默认值**: `0`
- **含义**: 如果设置为 > 0 的整数，出现在 `encoder_input_ids` 中的该大小的所有 ngram 不能出现在 `decoder_input_ids` 中

#### `decoder_start_token_id`
- **类型**: `int` 或 `list[int]` (可选)
- **默认值**: `None`
- **含义**: 如果编码器-解码器模型以不同于 *bos* 的 token 开始解码，该 token 的 id 或长度为 `batch_size` 的列表
- **说明**: 指示列表允许批次中不同元素使用不同的起始 id（例如，一个批次中不同目标语言的多语言模型）

### 8. 辅助生成参数

这些参数用于 assistant 模型和 speculative decoding。

#### `is_assistant`
- **类型**: `bool`
- **默认值**: `False`
- **含义**: 模型是否是 assistant（草稿）模型

#### `num_assistant_tokens`
- **类型**: `int`
- **默认值**: `20`
- **含义**: 定义在每次迭代中由 assistant 模型生成并受目标模型检查的*推测 token* 数量
- **说明**:
  - `num_assistant_tokens` 的值越高使生成更加*推测性*
  - 如果 assistant 模型性能好，可以达到更高的加速
  - 如果 assistant 模型需要大量修正，加速效果较低

#### `num_assistant_tokens_schedule`
- **类型**: `str`
- **默认值**: `"constant"`
- **含义**: 定义在推理期间更改最大 assistant tokens 的计划
- **可选值**:
  - `"heuristic"`: 当所有推测 token 都正确时，`num_assistant_tokens` 增加 2，否则减少 1。`num_assistant_tokens` 值在具有相同 assistant 模型的多个生成调用中持久存在
  - `"heuristic_transient"`: 与 `"heuristic"` 相同，但 `num_assistant_tokens` 在每次生成调用后重置为其初始值
  - `"constant"`: `num_assistant_tokens` 在生成期间保持不变

#### `assistant_confidence_threshold`
- **类型**: `float`
- **默认值**: `0.4`
- **含义**: assistant 模型的置信度阈值
- **说明**:
  - 如果 assistant 模型对当前 token 预测的置信度低于此阈值，assistant 模型停止当前 token 生成迭代，即使*推测 token* 数量（由 `num_assistant_tokens` 定义）尚未达到
  - assistant 的置信度阈值在整个推测迭代过程中进行调整，以减少不必要的草稿和目标前向传递，偏向于避免假阴性
  - `assistant_confidence_threshold` 值在具有相同 assistant 模型的多个生成调用中持久存在

#### `prompt_lookup_num_tokens`
- **类型**: `int` (可选)
- **默认值**: `None`
- **含义**: 要作为候选 token 输出的 token 数量

#### `max_matching_ngram_size`
- **类型**: `int` (可选)
- **默认值**: `None`
- **含义**: 考虑在提示中匹配的最大 ngram 大小
- **说明**: 如果未提供，默认为 2

#### `assistant_early_exit`
- **类型**: `int` (可选)
- **默认值**: `None`
- **含义**: 如果设置为正整数，将使用模型的提前退出作为 assistant
- **说明**: 只能用于支持提前退出的模型（即中间层的 logits 可以被 LM head 解释的模型）

#### `assistant_lookbehind`
- **类型**: `int` (可选)
- **默认值**: `10`
- **含义**: 如果设置为正整数，重新编码过程将额外考虑最后 `assistant_lookbehind` 个 assistant token 以正确对齐 token
- **说明**: 只能在 speculative decoding 中使用不同的分词器。参见 [这篇博客](https://huggingface.co/blog/universal_assisted_generation) 了解更多详情。

#### `target_lookbehind`
- **类型**: `int` (可选)
- **默认值**: `10`
- **含义**: 如果设置为正整数，重新编码过程将额外考虑最后 `target_lookbehind` 个目标 token 以正确对齐 token
- **说明**: 只能在 speculative decoding 中使用不同的分词器。参见 [这篇博客](https://huggingface.co/blog/universal_assisted_generation) 了解更多详情。

### 9. 性能优化参数

这些参数与性能和编译相关。

#### `compile_config`
- **类型**: `CompileConfig` (可选)
- **默认值**: `None`
- **含义**: 如果使用可编译缓存，这控制 `generate` 如何 `compile` 前向传递以实现更快推理

#### `disable_compile`
- **类型**: `bool` (可选)
- **默认值**: `False`
- **含义**: 是否禁用前向传递的自动编译
- **说明**: 当满足特定标准（包括使用可编译缓存）时会发生自动编译。如果发现需要使用此标志，请提出 issue。

## 使用示例

```python
from transformers import GenerationConfig

# 创建生成配置
generation_config = GenerationConfig(
    max_new_tokens=100,           # 最多生成 100 个 token
    min_new_tokens=10,            # 至少生成 10 个 token
    do_sample=True,               # 使用采样而不是贪心解码
    temperature=0.7,              # 温度参数，控制随机性
    top_p=0.9,                    # nucleus sampling
    top_k=50,                     # 只考虑概率最高的 50 个 token
    repetition_penalty=1.1,       # 重复惩罚
    num_beams=4,                  # 使用 beam search
    early_stopping=True,          # 早停
    no_repeat_ngram_size=2,       # 避免 2-gram 重复
    pad_token_id=0,               # 填充 token id
    eos_token_id=2,               # 结束 token id
)

# 保存配置
generation_config.save_pretrained("./my_model/")

# 从预训练模型加载配置
generation_config = GenerationConfig.from_pretrained("openai-community/gpt2")
```

## 生成模式

`GenerationConfig` 支持以下生成模式：

1. **贪心搜索** (`greedy_search`): `num_beams=1` 且 `do_sample=False`
2. **多项式采样** (`sample`): `num_beams=1` 且 `do_sample=True`
3. **Beam Search** (`beam_search`): `num_beams>1` 且 `do_sample=False`
4. **Beam Sample** (`beam_sample`): `num_beams>1` 且 `do_sample=True`
5. **约束 Beam Search** (`constrained_beam_search`): 设置了 `constraints` 或 `force_words_ids`
6. **Group Beam Search** (`group_beam_search`): `num_beam_groups>1`
7. **对比搜索** (`contrastive_search`): 设置了 `penalty_alpha` 和 `top_k`
8. **辅助生成** (`assisted_generation`): 设置了 `assistant_model` 或 `prompt_lookup_num_tokens`

每种模式都有其特定的参数要求和适用场景。

## 注意事项

1. **参数验证**: GenerationConfig 会在初始化时验证参数的有效性
2. **向后兼容**: 支持从模型的配置中自动继承生成参数
3. **保存和加载**: 可以保存到磁盘并从磁盘加载
4. **Hub 集成**: 支持推送到 Hugging Face Hub
5. **参数覆盖**: 在调用 `generate()` 时可以覆盖配置中的参数