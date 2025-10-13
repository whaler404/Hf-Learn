# 推理主流程详解

本文档详细分析 GenerationMixin 类中 `generate()` 方法的完整执行流程，这是大语言模型推理的核心入口点。

## 1. generate() 方法概览

`generate()` 方法位于 `generation/utils.py:2234`，是所有生成模型的统一入口点。它通过 9 个主要阶段完成从输入到输出的完整推理过程。

```python
@torch.no_grad()
def generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    # ... 其他参数
) -> Union[GenerateOutput, torch.LongTensor]:
```

## 2. 推理流程详细分析

### 阶段 0: 自定义生成函数处理 (2347-2367)

如果指定了 `custom_generate` 参数，系统会从 Hugging Face Hub 加载自定义的生成逻辑并执行，跳过标准流程。

```python
if custom_generate is not None and isinstance(custom_generate, str):
    custom_generate_function = self.load_custom_generate(custom_generate, trust_remote_code=trust_remote_code)
    return custom_generate_function(model=self, **generate_arguments)
```

### 阶段 1: 配置准备和验证 (2369-2411)

#### 1.1 提取生成模式相关参数
```python
generation_mode_kwargs = self._extract_generation_mode_kwargs(
    custom_generate, kwargs, synced_gpus, assistant_model, streamer
)
```

#### 1.2 准备生成配置
```python
generation_config, model_kwargs = self._prepare_generation_config(
    generation_config, use_model_defaults, **kwargs
)
```

#### 1.3 确定生成模式
```python
generation_mode = generation_config.get_generation_mode(assistant_model)
decoding_method = getattr(type(self), GENERATION_MODES_MAPPING[generation_mode])
```

**生成模式映射** (`generation/utils.py:132`):
- `GenerationMode.SAMPLE`: "_sample"
- `GenerationMode.GREEDY_SEARCH`: "_sample"
- `GenerationMode.BEAM_SEARCH`: "_beam_search"
- `GenerationMode.BEAM_SAMPLE`: "_beam_search"
- `GenerationMode.ASSISTED_GENERATION`: "_assisted_decoding"

### 阶段 2: 基础参数设置 (2413-2420)

```python
# 设置 logits 处理器和停止条件
logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

# 检查是否支持 attention mask
accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
```

### 阶段 3: 模型输入准备 (2421-2431)

#### 3.1 准备输入张量
```python
inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
    inputs, generation_config.bos_token_id, model_kwargs
)
```

#### 3.2 设置批次大小和设备
```python
batch_size = inputs_tensor.shape[0]
device = inputs_tensor.device
```

#### 3.3 准备特殊标记
```python
self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)
```

**特殊标记处理包括**:
- `bos_token_id`: 序列开始标记
- `eos_token_id`: 序列结束标记
- `pad_token_id`: 填充标记
- `decoder_start_token_id`: 解码器开始标记（编码器-解码器模型）

### 阶段 4: 模型参数准备 (2449-2487)

#### 4.1 缓存强制启用（解码器-only 模型）
```python
if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
    generation_config.use_cache = True
```

#### 4.2 准备注意力掩码
```python
if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
    model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
        inputs_tensor, generation_config, model_kwargs
    )
```

#### 4.3 编码器-解码器模型特殊处理
```python
if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
    model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
        inputs_tensor, model_kwargs, model_input_name, generation_config
    )
```

#### 4.4 准备解码器输入
```python
if self.config.is_encoder_decoder:
    input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(...)
else:
    input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")
```

#### 4.5 输入扩展（根据 beam 数量和返回序列数）
```python
input_ids, model_kwargs = self._expand_inputs_for_generation(
    input_ids=input_ids,
    expand_size=max(generation_config.num_beams, generation_config.num_return_sequences),
    is_encoder_decoder=self.config.is_encoder_decoder,
    **model_kwargs,
)
```

### 阶段 5: 长度参数准备 (2495-2515)

#### 5.1 计算输入长度并调整生成长度
```python
input_ids_length = input_ids.shape[1]
generation_config = self._prepare_generated_length(
    generation_config=generation_config,
    has_default_max_length=has_default_max_length,
    has_default_min_length=has_default_min_length,
    model_input_name=model_input_name,
    inputs_tensor=inputs_tensor,
    input_ids_length=input_ids_length,
)
```

#### 5.2 logits 优化
```python
if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
    model_kwargs["logits_to_keep"] = 1  # 只计算最后一个 token 的 logits，节省内存
```

### 阶段 6: 缓存初始化 (2516-2529)

```python
max_cache_length = generation_config.max_length - 1
self._prepare_cache_for_generation(
    generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
)
```

**缓存类型支持**:
- `"dynamic"`: DynamicCache - 默认选项
- `"static"`: StaticCache - 固定大小缓存
- `"offloaded"`: DynamicCache(offloaded=True) - CPU 卸载缓存
- `"quantized"`: QuantizedCache - 量化缓存

### 阶段 7: Logits 处理器和停止条件准备 (2542-2558)

#### 7.1 准备 logits 处理器
```python
prepared_logits_processor = self._get_logits_processor(
    generation_config=generation_config,
    input_ids_seq_length=input_ids_length,
    encoder_input_ids=inputs_tensor,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    logits_processor=logits_processor,
    device=inputs_tensor.device,
    model_kwargs=model_kwargs,
    negative_prompt_ids=negative_prompt_ids,
    negative_prompt_attention_mask=negative_prompt_attention_mask,
)
```

#### 7.2 准备停止条件
```python
prepared_stopping_criteria = self._get_stopping_criteria(
    generation_config=generation_config,
    stopping_criteria=stopping_criteria,
    tokenizer=generation_mode_kwargs.get("tokenizer"),
)
```

### 阶段 8: 执行生成循环 (2563-2572)

```python
model_kwargs["use_cache"] = generation_config.use_cache

result = decoding_method(
    self,
    input_ids,
    logits_processor=prepared_logits_processor,
    stopping_criteria=prepared_stopping_criteria,
    generation_config=generation_config,
    **generation_mode_kwargs,
    **model_kwargs,
)
```

**decoding_method** 根据生成模式选择具体实现：
- `_sample()`: 处理贪心搜索和采样
- `_beam_search()`: 处理 beam search
- `_assisted_decoding()`: 处理辅助生成

### 阶段 9: 缓存格式转换和返回 (2574-2581)

```python
# 转换为传统缓存格式（如果需要）
if generation_config.return_legacy_cache is True and hasattr(result, "past_key_values"):
    result.past_key_values = result.past_key_values.to_legacy_cache()

return result
```

## 3. 核心生成循环分析

### _sample() 方法详解

_sample() 方法 (`generation/utils.py:2686`) 是最基础的生成循环，同时支持贪心搜索和多项式采样。

#### 3.1 初始化阶段 (2728-2778)

```python
# 获取基础配置
pad_token_id = generation_config._pad_token_tensor
do_sample = generation_config.do_sample
batch_size, cur_len = input_ids.shape[:2]

# 初始化未完成序列跟踪
unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

# 设置模型前向传播
model_forward = self.__call__
compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
if compile_forward:
    model_forward = self.get_compiled_call(generation_config.compile_config)
```

#### 3.2 主生成循环 (2779-2849)

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

    # 3. 更新模型参数（特别是缓存）
    model_kwargs = self._update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
    )

    # 4. 获取下一个 token 的 logits
    next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32)

    # 5. 应用 logits 处理器
    next_token_scores = logits_processor(input_ids, next_token_logits)

    # 6. Token 选择
    if do_sample:
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        next_tokens = torch.argmax(next_token_scores, dim=-1)

    # 7. 更新序列和状态
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)

    # 8. 清理内存
    del outputs
```
