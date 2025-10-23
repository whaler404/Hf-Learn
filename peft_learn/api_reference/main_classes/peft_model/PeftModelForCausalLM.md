# PeftModelForCausalLM 概述

## 类的描述

PeftModelForCausalLM 是专为因果语言建模（Causal Language Modeling）任务设计的 PEFT 模型类，继承自 PeftModel。它为自回归文本生成任务提供了特殊的适配器配置和处理逻辑，支持多种参数高效微调方法，包括 LoRA、Prefix Tuning、Prompt Tuning、CPT 等。该类针对不同的 PEFT 方法实现了专门的前向传播逻辑，确保在各种微调策略下都能正确处理序列生成任务。

## 核心方法

- [generate](#generate)
    - [prepare_inputs_for_generation](#prepare_inputs_for_generation)
    - [PeftModelForCausalLM.forward](#forward)
        - prompt_learning: [get_prompt](PeftModel.md#get_prompt)


## 类的参数

- **model** (`~transformers.PreTrainedModel`): 基础 transformer 模型
- **peft_config** (`PeftConfig`): PEFT 配置对象
- **adapter_name** (`str`, *可选*): 适配器的名称，默认为 `"default"`
- **autocast_adapter_dtype** (`bool`, *可选*): 是否自动转换适配器数据类型。默认为 `True`。目前只将 float16 和 bfloat16 的适配器权重转换为 float32，这通常是稳定训练所需的，只影响选定的 PEFT 调优器

## 使用案例

```python
from transformers import AutoModelForCausalLM
from peft import PeftModelForCausalLM, get_peft_config

config = {
    "peft_type": "PREFIX_TUNING",
    "task_type": "CAUSAL_LM",
    "inference_mode": False,
    "num_virtual_tokens": 20,
    "token_dim": 1280,
    "num_transformer_submodules": 1,
    "num_attention_heads": 20,
    "num_layers": 36,
    "encoder_hidden_size": 1280,
    "prefix_projection": False,
    "postprocess_past_key_value_function": None,
 }

peft_config = get_peft_config(config)
model = AutoModelForCausalLM.from_pretrained("gpt2-large")
peft_model = PeftModelForCausalLM(model, peft_config)
peft_model.print_trainable_parameters()
# trainable params: 1843200 || all params: 775873280 || trainable%: 0.23756456724479544
```

# forward

## 传入参数

```python
input_ids  # [batch_size, sequence_length] - 输入 token IDs
attention_mask  # [batch_size, sequence_length] - 注意力掩码
inputs_embeds  # [batch_size, sequence_length, hidden_size] - 输入嵌入
labels  # [batch_size, sequence_length] - 训练标签
output_attentions  # bool - 是否输出注意力权重
output_hidden_states  # bool - 是否输出隐藏状态
return_dict  # bool - 是否返回字典格式输出
task_ids  # [batch_size] - 任务 ID（用于多任务学习）
**kwargs  # 其他关键字参数
```

## method 分析

这是 PeftModelForCausalLM 的关键代码，涉及到不同 peft_type 下的处理逻辑。代码中主要是不同条件判断和分支执行，判断逻辑的顺序和层次关系如下：

```python
def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None,
           output_attentions=None, output_hidden_states=None, return_dict=None,
           task_ids=None, **kwargs):
    peft_config = self.active_peft_config

    # 第一个主要判断：是否为提示学习方法
    if not peft_config.is_prompt_learning:
        # 非提示学习方法的处理逻辑（LoRA、AdaLoRA 等）

        # 特殊处理：MPT 模型类型
        if self.base_model.config.model_type == "mpt":
            # MPT 模型不支持 inputs_embeds，直接返回基础模型输出
            if inputs_embeds is not None:
                raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
            return self.base_model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                return_dict=return_dict, **kwargs,
            )

        # POLY 类型特殊处理：添加任务 ID
        if peft_config.peft_type == PeftType.POLY:
            kwargs["task_ids"] = task_ids

        # 启用 PEFT 前向钩子并调用基础模型
        with self._enable_peft_forward_hooks(**kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
            return self.base_model(
                input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds,
                labels=labels, output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                return_dict=return_dict, **kwargs,
            )

    # 提示学习方法的处理逻辑
    batch_size = _get_batch_size(input_ids, inputs_embeds)  # [batch_size]

    # 处理注意力掩码：为提示添加前缀注意力掩码
    if attention_mask is not None:
        # 创建前缀注意力掩码，维度为 [batch_size, num_virtual_tokens]
        prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
        # 将前缀掩码与原掩码连接，得到 [batch_size, num_virtual_tokens + sequence_length]
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

    # 清理不支持的参数
    if kwargs.get("position_ids", None) is not None:
        warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
        kwargs["position_ids"] = None
    if kwargs.get("token_type_ids", None) is not None:
        warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
        kwargs["token_type_ids"] = None

    # 更新 kwargs
    kwargs.update({
        "attention_mask": attention_mask, "labels": labels,
        "output_attentions": output_attentions, "output_hidden_states": output_hidden_states,
        "return_dict": return_dict,
    })

    # 第二个主要判断：根据具体的 PEFT 类型处理
    if peft_config.peft_type == PeftType.PREFIX_TUNING:
        # PREFIX_TUNING 处理逻辑
        # 计算最大缓存长度，为虚拟提示预留空间
        if input_ids is not None:
            max_cache_len = input_ids.shape[1] + peft_config.num_virtual_tokens  # [sequence_length + num_virtual_tokens]
        else:
            max_cache_len = inputs_embeds.shape[1] + peft_config.num_virtual_tokens  # [sequence_length + num_virtual_tokens]

        # 获取前缀键值对，维度取决于具体模型架构
        kwargs["past_key_values"] = self.get_prompt(batch_size, max_cache_len=max_cache_len)
        # 直接调用基础模型，使用 past_key_values
        return self.base_model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)

    elif peft_config.peft_type == PeftType.CPT:
        # CPT (Continuous Prompt Tuning) 处理逻辑
        # 调用专门的 CPT 前向传播方法
        return self._cpt_forward(input_ids, inputs_embeds, peft_config, task_ids, batch_size, **kwargs)

    else:
        # 其他提示学习方法（PROMPT_TUNING, P_TUNING, MULTITASK_PROMPT_TUNING 等）
        # 如果没有提供输入嵌入，则从 input_ids 生成
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)  # [batch_size, sequence_length, hidden_size]

        # 处理标签：为提示部分添加填充标签（-100）
        if labels is not None:
            # 创建前缀标签，全部为 -100（表示在计算损失时忽略）
            prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
            # 连接前缀标签和原标签，维度为 [batch_size, num_virtual_tokens + sequence_length]
            kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)

        # 获取提示嵌入
        prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)  # [batch_size, num_virtual_tokens, hidden_size]
        prompts = prompts.to(inputs_embeds.dtype)

        # 将提示嵌入与输入嵌入连接
        # [batch_size, num_virtual_tokens, hidden_size] + [batch_size, sequence_length, hidden_size]
        # -> [batch_size, num_virtual_tokens + sequence_length, hidden_size]
        inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)

        # 调用基础模型，使用连接后的嵌入
        return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
```

## _cpt_forward 方法分析

CPT 方法的特殊前向传播处理逻辑：

```python
def _cpt_forward(self, input_ids, inputs_embeds, peft_config, task_ids, batch_size, **kwargs):
    # 提取并处理标签
    labels = kwargs.pop("labels")  # [batch_size, sequence_length]
    device = [i.device for i in [input_ids, inputs_embeds, labels] if i is not None][0]

    # 处理输入类型掩码
    if "input_type_mask" in kwargs.keys():
        input_type_mask = kwargs.pop("input_type_mask").to(device)
    else:
        # 如果没有提供 input_type_mask，创建默认值
        if input_ids is None:
            N_tokens = inputs_embeds.shape[1]  # sequence_length
        else:
            N_tokens = input_ids.shape[1]  # sequence_length
        input_type_mask = torch.ones((batch_size, N_tokens)).to(device) * 4  # [batch_size, sequence_length]

    # 获取 CPT 配置
    cpt_token_ids = peft_config.cpt_token_ids
    cpt_tokens_type_mask = peft_config.cpt_tokens_type_mask

    # 生成输入嵌入（如果未提供）
    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)  # [batch_size, sequence_length, hidden_size]

    # 获取并连接提示嵌入
    prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)  # [batch_size, num_virtual_tokens, hidden_size]
    prompts = prompts.to(inputs_embeds.dtype)
    inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)  # [batch_size, total_length, hidden_size]

    # 处理标签（如果提供）
    cpt_labels = None
    if labels is not None:
        # 生成前缀标签
        prefix_labels = torch.Tensor(cpt_token_ids).long().view(1, -1)  # [1, num_virtual_tokens]
        prefix_labels = prefix_labels.repeat(batch_size, 1).to(labels.device)  # [batch_size, num_virtual_tokens]
        cpt_labels = torch.cat((prefix_labels, labels), dim=1)  # [batch_size, total_length]

        # 处理类型掩码
        prefix_type_mask = torch.Tensor(cpt_tokens_type_mask).long().view(1, -1)  # [1, num_virtual_tokens]
        prefix_type_mask = prefix_type_mask.repeat(batch_size, 1).to(labels.device)  # [batch_size, num_virtual_tokens]

        # 调整输入类型掩码以避免冲突
        adjusted_input_type_mask = input_type_mask  # [batch_size, sequence_length]
        adjusted_input_type_mask[adjusted_input_type_mask > 0] += prefix_type_mask.max()

        # 连接类型掩码
        cpt_type_mask = torch.cat((prefix_type_mask, adjusted_input_type_mask), dim=1)  # [batch_size, total_length]

        # 标识有效标签位置，无效位置用 -100 掩码
        labels_idx = (cpt_type_mask > 0) & (cpt_type_mask % 4 == 0)  # [batch_size, total_length]
        cpt_labels[~labels_idx] = -100

    kwargs["labels"] = cpt_labels

    # 调用基础模型
    base_model_output = self.base_model(inputs_embeds=inputs_embeds, **kwargs)
    if labels is None:
        return base_model_output
    else:
        # 使用自定义 CPT 损失函数计算损失
        cpt_embedding = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]
        base_model_output = cpt_embedding.calculate_loss(
            base_model_output, cpt_labels, cpt_type_mask, self.peft_config["default"]
        )
        return base_model_output
```

## 总结

PeftModelForCausalLM 的 forward 方法根据 PEFT 配置的类型采用不同的处理策略：

1. **非提示学习方法**（LoRA、AdaLoRA 等）：直接使用基础模型，可能启用特殊钩子
2. **PREFIX_TUNING**：通过 past_key_values 注入前缀信息
3. **CPT**：使用专门的 CPT 处理逻辑，包括特殊的标签和类型掩码处理
4. **其他提示学习方法**：将提示嵌入直接连接到输入嵌入中

每种方法都有其特定的张量操作和参数处理逻辑，以确保在相应的 PEFT 策略下能够正确进行前向传播和训练。

# generate

`generate` 方法是 PeftModelForCausalLM 中用于文本生成的核心方法。它为不同的 PEFT 类型提供了专门的生成处理逻辑，确保在生成过程中能够正确应用适配器权重和提示信息。

## method 分析

```python
def generate(self, *args, **kwargs):
    peft_config = self.active_peft_config

    # 设置生成配置
    self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
    if hasattr(self.base_model, "model"):
        self.base_model.model.generation_config = self.generation_config
    else:
        self.base_model.generation_config = self.generation_config

    try:
        # 第一个条件判断：是否为提示学习方法
        if not peft_config.is_prompt_learning:
            # 非提示学习方法的处理逻辑（LoRA、AdaLoRA 等）
            with self._enable_peft_forward_hooks(*args, **kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                outputs = self.base_model.generate(*args, **kwargs)
        else:
            # 提示学习方法的处理逻辑
            outputs = self.base_model.generate(**kwargs)
    except:
        # 异常处理：恢复原始的生成配置
        self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
        raise
    else:
        # 正常完成：恢复原始的生成配置
        self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
        return outputs
```

## 生成流程分析

1. **非提示学习方法**：启用 PEFT 前向钩子，过滤特殊参数，直接调用基础模型生成
2. **提示学习方法**：直接调用基础模型生成，依赖 `prepare_inputs_for_generation` 处理提示注入
3. **异常恢复**：确保在任何情况下都能正确恢复原始生成配置

# prepare_inputs_for_generation

`prepare_inputs_for_generation` 方法是为生成过程准备输入的核心方法。它根据不同的 PEFT 类型和生成阶段，动态调整输入参数，包括缓存处理、注意力掩码调整、提示注入等。

## 参数准备总结

该方法主要准备以下参数：

### past_key_values

**条件**：当 `peft_config.peft_type == PeftType.PREFIX_TUNING` 且需要提示注入时

```python
# 检查是否需要提示注入：没有 past_key_values 或缓存为空
requires_prompt_injection = (model_kwargs.get("past_key_values", None) is None) or (
    isinstance(model_kwargs["past_key_values"], transformers.Cache)
    and not model_kwargs["past_key_values"].get_seq_length()
)

if requires_prompt_injection and peft_config.peft_type == PeftType.PREFIX_TUNING:
    # 获取当前 past_key_values 的最大缓存长度
    max_cache_len = getattr(model_kwargs.get("past_key_values", None), "max_cache_len", None)

    # 获取前缀键值对：[num_layers, 2, batch_size, num_heads, head_dim]
    new_past_key_values = self.get_prompt(
        batch_size=model_kwargs["input_ids"].shape[0],  # batch_size
        max_cache_len=max_cache_len,
    )

    # 更新 past_key_values
    model_kwargs["past_key_values"] = new_past_key_values
```

### attention_mask

**条件**：当使用提示学习方法且存在注意力掩码时

```python
if (attention_mask := model_kwargs.get("attention_mask", None)) is not None:
    # 处理字典格式的注意力掩码（新版本 transformers 格式）
    if isinstance(attention_mask, dict):
        if len(attention_mask) != 1:
            raise ValueError(f"Expected a single attention mask, got {len(attention_mask)} instead")
        attention_mask = list(attention_mask.values())[0]

    # 创建前缀注意力掩码：[batch_size, num_virtual_tokens]
    size = model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
    prefix_attention_mask = torch.ones(size).to(model_kwargs["input_ids"].device)

    if attention_mask.dim() == 4:
        # 处理 4D 注意力掩码：[batch_size, heads, input_ids_length, total_sequence_length]
        bs = attention_mask.shape[0]
        total_seq_len = prefix_attention_mask.shape[1] + attention_mask.shape[2]

        # 转换为 2D 注意力掩码：[batch_size, total_sequence_length]
        attention_mask_2d = torch.ones((bs, total_seq_len), dtype=attention_mask.dtype)

        # 处理缓存位置
        if is_prefill and (peft_config.peft_type != PeftType.PREFIX_TUNING):
            # 在预填充阶段，为非 Prefix Tuning 方法设置缓存位置
            cache_position_ = torch.arange(total_seq_len, device=model_kwargs["input_ids"].device)
        else:
            # Prefix Tuning 直接作用于缓存，无需更新缓存位置
            cache_position_ = model_kwargs["cache_position"]

        # 创建新的注意力掩码
        attention_mask_new = create_attention_mask(
            self.get_base_model(),
            model_input=None,
            attention_mask=attention_mask_2d,
            past_key_values=model_kwargs.get("past_key_values"),
            cache_position=cache_position_,
            batch_size=bs,
            sequence_length=total_seq_len,
            position_ids=model_kwargs.get("position_ids", None),
        )
        model_kwargs["attention_mask"] = attention_mask_new
    else:
        # 处理 2D 注意力掩码：[batch_size, sequence_length]
        # 连接前缀掩码和原掩码：[batch_size, num_virtual_tokens + sequence_length]
        model_kwargs["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)
```

### inputs_embeds

**条件**：当需要提示注入且不是 PREFIX_TUNING 时

```python
elif requires_prompt_injection:
    # 非PREFIX_TUNING 的提示学习方法需要注入提示嵌入
    # 生成输入嵌入：[batch_size, sequence_length, hidden_size]
    inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])

    # 获取提示嵌入：[batch_size, num_virtual_tokens, hidden_size]
    prompts = self.get_prompt(
        batch_size=model_kwargs["input_ids"].shape[0],
        task_ids=task_ids
    )
    prompts = prompts.to(inputs_embeds.dtype)

    # 连接提示嵌入和输入嵌入：[batch_size, total_length, hidden_size]
    model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)

    # 清除 input_ids，使用 inputs_embeds
    model_kwargs["input_ids"] = None
```

### input_ids

**条件**：在提示学习的自回归生成阶段，当缓存长度超过输入长度时

```python
if uses_cache and (model_kwargs.get("past_key_values", None) is not None):
    past_key_values = model_kwargs["past_key_values"]

    # 获取当前序列长度
    if isinstance(past_key_values, (tuple, list)):
        seq_len = past_key_values[0][0].shape[-2]  # [sequence_length]
    else:  # 使用 transformers KV cache
        seq_len = past_key_values.get_seq_length()

    # 如果缓存长度 >= 输入长度，只保留最后一个 input_id（自回归生成）
    if seq_len >= model_kwargs["input_ids"].shape[1]:
        model_kwargs["input_ids"] = model_kwargs["input_ids"][:, -1:]  # [batch_size, 1]
```

### cache_position

**条件**：在不同 PEFT 类型和生成阶段下的特殊处理

```python
# 如果在预填充阶段且为 PREFIX_TUNING
if is_prefill and (peft_config.peft_type == PeftType.PREFIX_TUNING):
    # PREFIX_TUNING 的 past_key_values 已预填充，需要调整缓存位置
    model_kwargs["cache_position"] += peft_config.num_virtual_tokens
elif peft_config.peft_type != PeftType.PREFIX_TUNING:
    # 非 PREFIX_TUNING 方法移除缓存位置，让模型重新创建
    _ = model_kwargs.pop("cache_position", None)
```

### task_ids

**条件**：当 `peft_config.peft_type == PeftType.POLY` 时

```python
if peft_config.peft_type == PeftType.POLY:
    # POLY 类型需要任务 ID
    model_kwargs["task_ids"] = task_ids
```


`prepare_inputs_for_generation` 方法通过以下步骤确保生成过程的正确性：

1. **生成阶段判断**：通过 `cache_position` 判断是否在预填充阶段
2. **提示注入**：在适当时机注入提示信息到输入中
3. **缓存管理**：正确处理 KV 缓存的位置和长度