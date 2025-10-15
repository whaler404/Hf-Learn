# prepare_inputs_for_generation 方法详细分析

## 概述

`prepare_inputs_for_generation` 方法是 Transformers 库中用于为 LLM 生成准备输入的核心方法。该方法位于 `generation/utils.py:546-687`，负责处理和转换输入张量，确保它们符合模型前向传播的期望格式。

## 方法签名

```python
def prepare_inputs_for_generation(
    self,
    input_ids: torch.LongTensor,
    past_key_values: Optional[Cache] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> dict[str, torch.Tensor]
```

- `input_ids`: 形状为 `(batch_size, sequence_length)` 的 LongTensor，包含输入序列的 token ID
- `past_key_values`: 可选的 Cache 对象，包含已计算的键值对缓存
- `attention_mask`: 可选的注意力掩码，形状为 `(batch_size, sequence_length)`
- `inputs_embeds`: 可选的嵌入表示，形状为 `(batch_size, sequence_length, hidden_size)`
- `cache_position`: 可选的缓存位置指示器，形状为 `(query_length,)`
- `**kwargs`: 其他模型特定的参数

## 详细处理流程

### 步骤 1: 初始化 model_inputs (lines 564-566)

```python
model_inputs = {}
model_inputs["cache_position"] = cache_position
```

- **操作**: 创建空的 `model_inputs` 字典，添加 `cache_position`
- **张量形状**: `cache_position` 保持原始形状 `(query_length,)`
- **内容**: 当前生成步骤的位置索引

### 步骤 2: 处理缓存相关的输入准备 (lines 568-576)

```python
if past_key_values is not None:
    model_inputs["past_key_values"] = past_key_values
    inputs_embeds, input_ids = self._cache_dependant_input_preparation(
        input_ids, inputs_embeds, cache_position
    )
```

当存在缓存时：

**2.1 添加 past_key_values**
- **操作**: 直接添加缓存到 model_inputs
- **内容**: 包含历史键值对的 Cache 对象

**2.2 缓存依赖的输入准备**
调用 `_cache_dependant_input_preparation` 方法处理 `input_ids` 和 `inputs_embeds`：

```python
def _cache_dependant_input_preparation(self, input_ids, inputs_embeds, cache_position):
    if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
        inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
    elif inputs_embeds is not None or (cache_position[-1] >= input_ids.shape[1]):  # Exception 1 & 3
        input_ids = input_ids[:, -cache_position.shape[0] :]
    elif input_ids.shape[1] != cache_position.shape[0]:  # Default case
        input_ids = input_ids[:, cache_position]
    return inputs_embeds, input_ids
```

**张量变换说明**:
- `input_ids.shape[1]`: 原始序列长度
- `cache_position.shape[0]`: 当前需要处理的 token 数量
- `input_ids[:, -cache_position.shape[0]:]`: 取最后 N 个 token，形状从 `(batch_size, seq_len)` 变为 `(batch_size, N)`
- `input_ids[:, cache_position]`: 根据位置索引选择 token，形状从 `(batch_size, seq_len)` 变为 `(batch_size, cache_position.shape[0])`

### 步骤 3: 准备基础模型输入 (lines 578-590)

```python
input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
if not self.config.is_encoder_decoder:
    if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
        model_inputs[input_ids_key] = None
        model_inputs["inputs_embeds"] = inputs_embeds
    else:
        model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
        model_inputs["inputs_embeds"] = None
else:
    model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
```

**3.1 确定输入键名**
- **操作**: 根据模型类型选择合适的键名
- **结果**:
  - encoder-decoder 模型: `"decoder_input_ids"`
  - decoder-only 模型: `"input_ids"`

**3.2 处理 inputs_embeds 和 input_ids**
- **条件判断**: `len(cache_position) == inputs_embeds.shape[1]`
- **True 时**: 使用 `inputs_embeds`，形状为 `(batch_size, query_length, hidden_size)`
- **False 时**: 使用 `input_ids`，形状为 `(batch_size, query_length)`
- **clone 操作**: `input_ids.clone(memory_format=torch.contiguous_format)` 确保内存连续性

### 步骤 4: 创建缺失的 position_ids (lines 592-606)

```python
encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None
attention_mask = (
    kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
)
attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"

if (
    attention_mask is not None
    and kwargs.get(position_ids_key) is None
    and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    kwargs[position_ids_key] = position_ids
```

**4.1 处理注意力掩码键名**
- **操作**: 根据模型类型设置正确的键名
- **结果**: encoder-decoder 模型使用 decoder 相关键名

**4.2 创建 position_ids**
当 `attention_mask` 存在且 `position_ids` 不存在时：

```python
position_ids = attention_mask.long().cumsum(-1) - 1
position_ids.masked_fill_(attention_mask == 0, 1)
```

**张量变换详解**:
1. `attention_mask.long()`: 将 attention_mask 转换为 long 类型，形状保持 `(batch_size, sequence_length)`
2. `cumsum(-1)`: 沿最后一个维度（序列长度）累积求和
   - 原始: `[[1, 1, 1, 0, 0]]` → cumsum 后: `[[1, 2, 3, 3, 3]]`
   - 减 1 后: `[[0, 1, 2, 2, 2]]`
3. `masked_fill_(attention_mask == 0, 1)`: 将 padding 位置（原值为0）的位置设为1
   - 结果: `[[0, 1, 2, 1, 1]]`
   - **最终形状**: `(batch_size, sequence_length)`

### 步骤 5: 切片需要与 input_ids 长度相同的模型输入 (lines 608-620)

```python
for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
    model_input = kwargs.get(model_input_name)
    if model_input is not None:
        if past_key_values is not None:
            current_input_length = (
                model_inputs["inputs_embeds"].shape[1]
                if model_inputs.get("inputs_embeds") is not None
                else model_inputs[input_ids_key].shape[1]
            )
            model_input = model_input[:, -current_input_length:]
            model_input = model_input.clone(memory_format=torch.contiguous_format)
        model_inputs[model_input_name] = model_input
```

**处理流程**:
1. **遍历需要处理的输入**: `position_ids`, `token_type_ids`, `decoder_position_ids`
2. **获取当前输入长度**:
   - 如果使用 `inputs_embeds`: `model_inputs["inputs_embeds"].shape[1]`
   - 如果使用 `input_ids`: `model_inputs[input_ids_key].shape[1]`
3. **切片操作**: `model_input[:, -current_input_length:]`
   - **原形状**: `(batch_size, original_sequence_length)`
   - **切片后形状**: `(batch_size, current_input_length)`
4. **内存连续性**: `clone(memory_format=torch.contiguous_format)` 确保张量内存连续

### 步骤 6: 创建 4D 注意力掩码 (lines 622-674)

这是最复杂的步骤，处理可编译缓存的注意力掩码转换：

```python
if (
    isinstance(past_key_values, Cache)
    and past_key_values.is_compileable
    and attention_mask is not None
    and attention_mask.ndim == 2
):
```

**6.1 获取批次和序列长度**
```python
if not self.config.is_encoder_decoder and model_inputs["inputs_embeds"] is not None:
    batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
else:
    batch_size, sequence_length = model_inputs[input_ids_key].shape[:2]
```

**形状信息**:
- 使用 `inputs_embeds`: `(batch_size, sequence_length, hidden_size)`
- 使用 `input_ids`: `(batch_size, sequence_length)`

**6.2 寻找因果掩码创建函数**
```python
base_model = getattr(self, self.base_model_prefix, self)
decoder = base_model.get_decoder() if hasattr(base_model, "get_decoder") else None
causal_mask_creation_function = getattr(
    base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
)
if causal_mask_creation_function is None and decoder is not None:
    causal_mask_creation_function = getattr(
        decoder, "_prepare_4d_causal_attention_mask_with_cache_position", None
    )
```

**6.3 创建 4D 掩码**
如果找不到专用函数，使用通用掩码创建：

```python
if causal_mask_creation_function is None:
    token_type_ids = model_inputs.get("token_type_ids")
    position_ids = model_inputs.get(position_ids_key)
    causal_mask_creation_function = getattr(self, "create_masks_for_generate", create_masks_for_generate)
    attention_mask = causal_mask_creation_function(
        config=self.config,
        input_embeds=torch.empty((batch_size, sequence_length), dtype=self.dtype),
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
        token_type_ids=token_type_ids,
    )
```

**张量变换**:
- **输入 attention_mask**: `(batch_size, sequence_length)` - 2D 掩码
- **输出 attention_mask**: `(batch_size, 1, query_length, kv_length)` - 4D 掩码
- **创建空嵌入**: `torch.empty((batch_size, sequence_length), dtype=self.dtype)` 仅用于推断形状

### 步骤 7: 添加注意力掩码到 model_inputs (lines 674-678)

```python
if attention_mask is not None:
    model_inputs[attention_mask_key] = attention_mask

if encoder_attention_mask is not None:
    model_inputs["attention_mask"] = encoder_attention_mask
```

**添加结果**:
- **decoder-only 模型**: `model_inputs["attention_mask"] = 4D_mask`
- **encoder-decoder 模型**:
  - `model_inputs["decoder_attention_mask"] = 4D_mask`
  - `model_inputs["attention_mask"] = encoder_attention_mask`

### 步骤 8: 转发所有未初始化的 kwargs (lines 680-683)

```python
for key, value in kwargs.items():
    if key not in model_inputs:
        model_inputs[key] = value
```

**操作**: 将所有未在 model_inputs 中的 kwargs 参数直接添加到最终输出中

### 步骤 9: 移除意外的生成输入 (lines 685-686)

```python
model_inputs.pop("labels", None)
return model_inputs
```

**操作**: 移除 "labels" 参数，因为它不应该出现在生成输入中

## 最终输出的 model_inputs

根据不同的配置和输入情况，`model_inputs` 字典可能包含以下键值对：

| Key 名称 | 条件性 | 数据类型 | 张量形状 | 说明 |
|----------|--------|----------|----------|------|
| `cache_position` | 总是存在 | torch.LongTensor | `(query_length,)` | 当前生成步骤的位置索引 |
| `past_key_values` | 当 `past_key_values` 参数不为 None 时 | Cache 对象 | - | 包含历史键值对的缓存 |
| `input_ids` | decoder-only 模型且不使用 `inputs_embeds` | torch.LongTensor | `(batch_size, query_length)` | 根据缓存切片后的输入 ID |
| `decoder_input_ids` | encoder-decoder 模型 | torch.LongTensor | `(batch_size, query_length)` | 解码器输入 ID |
| `inputs_embeds` | 使用嵌入表示时 | torch.FloatTensor | `(batch_size, query_length, hidden_size)` | 输入嵌入表示 |
| `position_ids` | decoder-only 模型且需要自动生成时 | torch.LongTensor | `(batch_size, query_length)` | 位置编码 |
| `decoder_position_ids` | encoder-decoder 模型且需要自动生成时 | torch.LongTensor | `(batch_size, query_length)` | 解码器位置编码 |
| `token_type_ids` | 当传入 `token_type_ids` 参数时 | torch.LongTensor | `(batch_size, query_length)` | token 类型 ID |
| `attention_mask` | decoder-only 模型 | torch.Tensor | `(batch_size, 1, query_length, kv_length)` | 4D 注意力掩码 |
| `decoder_attention_mask` | encoder-decoder 模型 | torch.Tensor | `(batch_size, 1, query_length, kv_length)` | 解码器 4D 注意力掩码 |
| `attention_mask` (encoder) | encoder-decoder 模型 | torch.LongTensor | `(batch_size, encoder_seq_length)` | 编码器注意力掩码 |
| 其他 kwargs | 当传入其他参数时 | varies | varies | 如 `use_cache`, `output_attentions` 等 |

这个方法是 LLM 生成的关键预处理步骤，确保所有输入张量都具有正确的形状和格式，以便高效的前向传播。