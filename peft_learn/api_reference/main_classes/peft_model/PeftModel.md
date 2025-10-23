# PeftModel 概述

PeftModel 是 PEFT（Parameter-Efficient Fine-Tuning）库的核心模型类，它封装了各种参数高效微调方法，为不同的预训练模型提供统一的接口。PeftModel 继承自 `PushToHubMixin` 和 `torch.nn.Module`，支持多种微调技术，如 LoRA、Prefix Tuning、Prompt Tuning 等。

## 核心方法

- [PeftModel.\_\_init\_\_](#__init__)
  - prompt learning: 
    - [PeftModel.add_adapter](#add_adapter)
      - [PeftModel._setup_prompt_encoder](#_setup_prompt_encoder)
        - [PrefixEncoder.\_\_init\_\_](../../adapters/prefix-tuning/PrefixEncoder.md#__init__)
  - lora
    - [BaseTuner.\_\_init\_\_](../tuners/BaseTuner.md#init)
- [PeftModel.forward](#forward)
  - [get_base_model](#get_base_model)
    - prompt learning: [PrefixEncoder.forward](../../adapters/prefix-tuning/PrefixEncoder.md#forward)
    - lora: [Linear.forward](../../adapters/lora/Linear.md#forward)

## 类的描述

PeftModel 是一个基础模型，包含各种 PEFT 方法。它为预训练模型添加参数高效的适配器，使得在保持大部分参数冻结的情况下，只训练少量参数就能适应新任务。该类支持多种 PEFT 技术，并提供了适配器管理、模型保存/加载、参数统计等功能。

## 类的参数

- **model** (`~transformers.PreTrainedModel`): 用于 PEFT 的基础 transformer 模型
- **peft_config** (`PeftConfig`): PEFT 模型的配置对象
- **adapter_name** (`str`, *可选*): 适配器的名称，默认为 `"default"`
- **autocast_adapter_dtype** (`bool`, *可选*): 是否自动转换适配器数据类型。默认为 `True`。目前只将 float16 和 bfloat16 的适配器权重转换为 float32，这通常是稳定训练所需的，只影响选定的 PEFT 调优器
- **low_cpu_mem_usage** (`bool`, *可选*, 默认为 `False`): 在 meta 设备上创建空的适配器权重。用于加速加载过程

## 属性

- **base_model** (`torch.nn.Module`): 用于 PEFT 的基础 transformer 模型
- **peft_config** (`PeftConfig`): PEFT 模型的配置对象
- **modules_to_save** (`list` of `str`): 保存模型时要保存的子模块名称列表
- **prompt_encoder** (`PromptEncoder`): 如果使用 [`PromptLearningConfig`]，则包含用于 PEFT 的提示编码器
- **prompt_tokens** (`torch.Tensor`): 如果使用 [`PromptLearningConfig`]，则包含用于 PEFT 的虚拟提示标记
- **transformer_backbone_name** (`str`): 如果使用 [`PromptLearningConfig`]，则包含基础模型中 transformer 主干的名称
- **word_embeddings** (`torch.nn.Embedding`): 如果使用 [`PromptLearningConfig`]，则包含基础模型中 transformer 主干的词嵌入

# 方法


## 初始化和配置方法

### `__init__`
- **方法描述**：初始化 PeftModel 实例
- **传入参数**：
  - `model` (`PreTrainedModel`): 基础 transformer 模型
  - `peft_config` (`PeftConfig`): PEFT 配置对象
  - `adapter_name` (`str`, 默认 `"default"`): 适配器名称
  - `autocast_adapter_dtype` (`bool`, 默认 `True`): 是否自动转换适配器数据类型
  - `low_cpu_mem_usage` (`bool`, 默认 `False`): 是否使用低 CPU 内存
- **返回参数**：无

#### method 解读
```python
# 调用父类初始化方法
super().__init__()

# 设置当前活动的适配器名称
self.active_adapter = adapter_name

# 从配置中获取 PEFT 类型（如 LoRA、Prefix Tuning 等）
self.peft_type = peft_config.peft_type

# 定义特殊的前向传播参数，这些参数需要从用户传入的参数中移除
self.special_peft_forward_args = {"adapter_names"}

# 检查是否为提示学习方法（如 Prompt Tuning、P-Tuning 等）
self._is_prompt_learning = peft_config.is_prompt_learning

# 根据是否为提示学习采用不同的初始化策略
if self._is_prompt_learning:
    # 提示学习方法：直接使用原始模型，并添加适配器
    # 初始化适配器配置字典，以适配器名称为键
    self._peft_config = {adapter_name: peft_config}
    # 保存对基础模型的引用
    self.base_model = model
    # 向模型添加适配器（配置为提示学习类型）
    self.add_adapter(adapter_name, peft_config, low_cpu_mem_usage=low_cpu_mem_usage)
else:
    # 参数高效微调方法（如 LoRA、AdaLoRA 等）：使用专门的调优器包装模型
    self._peft_config = None
    # 根据 PEFT 类型获取对应的调优器类（如 LoraModel、AdaLoraModel 等）
    # PeftModel >> BaseTuner
    cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]
    # 根据是否使用低 CPU 内存来选择上下文管理器
    ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
    with ctx():
        # 使用调优器包装基础模型，传入适配器配置映射和适配器名称
        self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)

# 如果基础模型支持数据类型转换，则配置适配器的数据类型自动转换
if hasattr(self.base_model, "_cast_adapter_dtype"):
    self.base_model._cast_adapter_dtype(
        adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype
    )

# 如果模型启用了梯度检查点，则准备模型以支持梯度检查点
if getattr(model, "is_gradient_checkpointing", True):
    model = self.prepare_model_for_gradient_checkpointing(model)

# 为了避免数值差异和意外行为，禁用预训练时的张量并行模拟
# 这是为了解决 Pytorch 的一个已知问题：https://github.com/pytorch/pytorch/issues/76232
if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
    self.base_model.config.pretraining_tp = 1
```

### `add_adapter`
- **方法描述**：根据传入的配置向模型添加适配器。此适配器未经过训练。要加载训练好的适配器，请使用 [`PeftModel.load_adapter`]。新适配器的名称应该是唯一的。新适配器不会自动设置为活动适配器。
- **传入参数**：
  - `adapter_name` (`str`): 要添加的适配器名称
  - `peft_config` (`PeftConfig`): 要添加的适配器配置
  - `low_cpu_mem_usage` (`bool`, *可选*, 默认 `False`): 在 meta 设备上创建空的适配器权重。用于加速加载保存适配器的过程。创建新的 PEFT 适配器进行训练时不要使用此选项
- **返回参数**：无

#### method 解读
```python
# 根据适配器类型获取对应的前缀映射，用于检查适配器名称是否合规
prefix = PEFT_TYPE_TO_PREFIX_MAPPING.get(peft_config.peft_type)

# 检查适配器名称是否包含在类型前缀中，如果包含则发出警告
# 这可能会导致加载时适配器权重的重新初始化
if prefix and adapter_name in prefix:
    warnings.warn(
        f"Adapter name {adapter_name} should not be contained in the prefix {prefix}."
        "This may lead to reinitialization of the adapter weights during loading."
    )

# 检查新适配器的类型是否与当前模型的 PEFT 类型一致
# 不允许在同一个模型中混合不同类型的适配器（如 LoRA 和 Prefix Tuning）
if peft_config.peft_type != self.peft_type:
    raise ValueError(
        f"Cannot combine adapters with different peft types. "
        f"Found {self.peft_type} and {peft_config.peft_type}."
    )

try:
    # 根据适配器类型采用不同的添加策略
    if peft_config.is_prompt_learning:
        # 提示学习方法（如 Prompt Tuning、P-Tuning 等）
        # 将适配器配置添加到配置字典中
        self.peft_config[adapter_name] = peft_config

        # 获取模型配置的字典表示
        if hasattr(self.config, "to_dict"):
            dict_config = self.config.to_dict()
        else:
            dict_config = self.config

        # 准备提示学习配置，确保与模型配置兼容
        peft_config = _prepare_prompt_learning_config(peft_config, dict_config)

        # 设置提示编码器，处理提示的表示和学习
        self._setup_prompt_encoder(adapter_name)

        # 设置额外的可训练模块（如特定的层或参数）
        set_additional_trainable_modules(
            model=self.base_model,
            peft_config=peft_config,
            model_config=BaseTuner.get_model_config(self),
            adapter_name=adapter_name,
        )
    elif peft_config.is_adaption_prompt:
        # 适配提示方法（Adaption Prompt）
        # 通过基础模型添加适配器
        self.base_model.add_adapter(adapter_name, peft_config)

        # 设置额外的可训练模块
        set_additional_trainable_modules(
            model=self.base_model,
            peft_config=peft_config,
            model_config=BaseTuner.get_model_config(self),
            adapter_name=adapter_name,
        )
    else:
        # 参数高效微调方法（如 LoRA、AdaLoRA 等）
        # 将适配器配置添加到配置字典中
        self.peft_config[adapter_name] = peft_config

        # 向基础模型注入适配器，这会在目标模块中添加适配器层
        self.base_model.inject_adapter(
            self.base_model.model, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage
        )
except Exception:  # 如果添加过程中出现错误，执行回滚操作
    # 从配置字典中移除已添加的适配器配置，保持模型状态一致性
    if adapter_name in self.peft_config:
        del self.peft_config[adapter_name]
    # 重新抛出异常，让调用者知道添加失败
    raise
```

### `delete_adapter`
- **方法描述**：删除现有适配器
- **传入参数**：
  - `adapter_name` (`str`): 要删除的适配器名称
- **返回参数**：无

### `set_adapter`
- **方法描述**：设置活动适配器。一次只能有一个适配器处于活动状态。此外，此函数将指定的适配器设置为可训练（即 requires_grad=True）。如果不需要这样，可以使用以下代码
- **传入参数**：
  - `adapter_name` (`str`): 要设置为活动适配器的适配器名称。适配器必须先加载
- **返回参数**：无

## 保存和加载方法

### `save_pretrained`
- **方法描述**：将适配器模型和适配器配置文件保存到目录，以便可以使用 [`PeftModel.from_pretrained`] 类方法重新加载，也可以被 [`PeftModel.push_to_hub`] 方法使用
- **传入参数**：
  - `save_directory` (`str`): 保存适配器模型和配置文件的目录（如果不存在将创建）
  - `safe_serialization` (`bool`, *可选*): 是否以 safetensors 格式保存适配器文件，默认为 `True`
  - `selected_adapters` (`List[str]`, *可选*): 要保存的适配器列表。如果为 `None`，将默认保存所有适配器
  - `save_embedding_layers` (`Union[bool, str]`, *可选*, 默认 `"auto"`): 如果为 `True`，除了适配器权重外还保存嵌入层。如果为 `"auto"`，在配置的 `target_modules` 中检查常见嵌入层 `peft.utils.other.EMBEDDING_LAYER_NAMES`，并自动设置布尔标志。这只适用于 🤗 transformers 模型
  - `is_main_process` (`bool`, *可选*): 调用此方法的进程是否为主进程。默认为 `True`。如果不是主进程则不保存检查点，这对于多设备设置（如 DDP）很重要
  - `path_initial_model_for_weight_conversion` (`str`, *可选*): 初始化适配器的路径，在用 PiSSA/CorDA/OLoRA 初始化模型后但在进行任何训练之前获得。当 `path_initial_model_for_weight_conversion` 不为 `None` 时，计算微调前后适配器的差异。这种差异可以表示为标准 LoRA 适配器的参数。使用此转换的适配器不需要更改基础模型，从而方便地允许多个 PiSSA/CorDA/OLoRA 适配器与 LoRA 适配器一起使用，以及任何适配器的激活或停用
  - `kwargs` (额外的关键字参数, *可选*): 传递给 `push_to_hub` 方法的额外关键字参数
- **返回参数**：无

### `from_pretrained`
- **方法描述**：从预训练模型和加载的 PEFT 权重实例化 PEFT 模型。注意传入的 `model` 可能会被就地修改
- **传入参数**：
  - `model` (`torch.nn.Module`): 要适配的模型。对于 🤗 Transformers 模型，模型应使用 [`~transformers.PreTrainedModel.from_pretrained`] 初始化
  - `model_id` (`str` 或 `os.PathLike`): 要使用的 PEFT 配置的名称。可以是：
    - 字符串，托管在 Hugging Face Hub 模型仓库内的 PEFT 配置的 `model id`
    - 包含使用 `save_pretrained` 方法保存的 PEFT 配置文件的目录路径（`./my_peft_config_directory/`）
  - `adapter_name` (`str`, *可选*, 默认 `"default"`): 要加载的适配器名称。这对于加载多个适配器很有用
  - `is_trainable` (`bool`, *可选*, 默认 `False`): 适配器是否应该是可训练的。如果为 `False`，适配器将被冻结，只能用于推理
  - `config` (`~peft.PeftConfig`, *可选*): 要使用的配置对象，而不是自动加载的配置。此配置对象与 `model_id` 和 `kwargs` 互斥。当配置在调用 `from_pretrained` 之前已经加载时很有用
  - `autocast_adapter_dtype` (`bool`, *可选*): 是否自动转换适配器数据类型。默认为 `True`。仅适用于特定适配器类型
  - `ephemeral_gpu_offload` (`bool`, *可选*): 是否对部分加载的模块使用临时 GPU 卸载。默认为 `False`。当模型和/或组件（如适配器）的部分在需要之前保持在 CPU 内存中时很有用
  - `low_cpu_mem_usage` (`bool`, *可选*, 默认 `False`): 在加载保存的权重之前在 meta 设备上创建空的适配器权重。用于加速过程
  - `torch_device` (`str`, *可选*, 默认 None): 加载适配器的设备。如果为 `None`，将推断设备
  - `key_mapping` (`dict`, *可选*, 默认 None): 在加载 `state_dict` 之前应用的 PEFT `state_dict` 键的额外映射。应用此映射时，会提前移除 PEFT 特定的 `"base_model.model"` 前缀，并且尚未插入适配器名称（例如 `"default"`）。只有在你了解自己在做什么时才传递此参数
  - `kwargs` (`可选`): 传递给特定 PEFT 配置类的额外关键字参数
- **返回参数**：`PeftModel` 实例

### `load_adapter`
- **方法描述**：将训练好的适配器加载到模型中。新适配器的名称应该是唯一的。新适配器不会自动设置为活动适配器。使用 [`PeftModel.set_adapter`] 设置活动适配器
- **传入参数**：
  - `model_id` (`str` 或 `os.PathLike`): 要使用的 PEFT 配置的名称。可以是：
    - 字符串，托管在 Hugging Face Hub 模型仓库内的 PEFT 配置的 `model id`
    - 包含使用 `save_pretrained` 方法保存的 PEFT 配置文件的目录路径（`./my_peft_config_directory/`）
  - `adapter_name` (`str`): 要添加的适配器名称
  - `is_trainable` (`bool`, *可选*, 默认 `False`): 适配器是否应该是可训练的。如果为 `False`，适配器将被冻结，只能用于推理
  - `torch_device` (`str`, *可选*, 默认 None): 加载适配器的设备。如果为 `None`，将推断设备
  - `autocast_adapter_dtype` (`bool`, *可选*, 默认 `True`): 是否自动转换适配器数据类型。默认为 `True`。现在这只将使用 float16 和 bfloat16 的适配器权重转换为 float32，这通常是稳定训练所需的，只影响选定的 PEFT 调优器
  - `ephemeral_gpu_offload` (`bool`, *可选*, 默认 `False`): 是否对部分加载的模块使用临时 GPU 卸载。默认为 `False`
  - `low_cpu_mem_usage` (`bool`, *可选*, 默认 `False`): 在加载保存的权重之前在 meta 设备上创建空的适配器权重。用于加速过程
  - `key_mapping` (`dict`, *可选*, 默认 None): 在加载 `state_dict` 之前应用的 PEFT `state_dict` 键的额外映射
  - `kwargs` (`可选`): 修改适配器加载方式的额外参数，例如 Hugging Face Hub 的令牌
- **返回参数**：加载结果

## 前向传播和生成方法

### `forward`
- **方法描述**：模型的前向传播
- **传入参数**：
  - `*args`: 位置参数
  - `**kwargs`: 关键字参数
- **返回参数**：模型的输出

### `generate`
- **方法描述**：生成序列
- **传入参数**：
  - `*args`: 位置参数
  - `**kwargs`: 关键字参数
- **返回参数**：生成的序列

### `prepare_inputs_for_generation`
- **方法描述**：为生成准备输入（仅适用于特定模型类）
- **传入参数**：
  - `*args`: 位置参数
  - `task_ids` (`torch.Tensor`, *可选*): 任务 ID
  - `**kwargs`: 关键字参数
- **返回参数**：准备好的模型输入

## 参数统计和状态查询方法

### `get_nb_trainable_parameters`
- **方法描述**：返回模型中可训练参数的数量和所有参数的数量
- **传入参数**：无
- **返回参数**：`tuple[int, int]` - (可训练参数数量, 所有参数数量)

### `print_trainable_parameters`
- **方法描述**：打印模型中可训练参数的数量。注意：print_trainable_parameters() 使用 get_nb_trainable_parameters()，这与来自 huggingface/transformers 的 num_parameters(only_trainable=True) 不同。get_nb_trainable_parameters() 返回包含修改的主干 transformer 模型的 Peft 模型的（可训练参数，所有参数）。对于像 LoRA 这样的技术，主干 transformer 模型被就地修改。然而，对于提示调优，主干 transformer 模型未被修改。num_parameters(only_trainable=True) 返回主干 transformer 模型的可训练参数数量，这可能不同
- **传入参数**：无
- **返回参数**：无

### `get_layer_status`
- **方法描述**：获取模型中每个适配器层的状态。此方法返回 `TunerLayerStatus` 数据类实例的列表，每个实例包含以下属性：
  - `name` (`str`): 适配器层的名称，例如 `model.encoder.block.0.layer.0.SelfAttention.q`
  - `module_type` (`str`): 适配器层的类型，例如 `lora.Linear`
  - `enabled` (`bool`): 适配器层是否启用
  - `active_adapters` (`list[str]`): 活动适配器的名称（如果有），例如 `["default"]`
  - `merged_adapters` (`list[str]`): 合并适配器的名称（如果有），例如 `["default"]`
  - `available_adapters` (`list[str]`): 可用适配器的名称，例如 `["default"]`
- **传入参数**：无
- **返回参数**：`list[TunerLayerStatus]` - 包含相应适配器层状态的数据类列表

### `get_model_status`
- **方法描述**：获取模型调优器的状态。此方法返回 `TunerModelStatus` 数据类实例，包含以下属性：
  - `base_model_type` (`str`): 基础模型的类型，例如 `T5Model`
  - `adapter_model_type` (`str`): 适配器模型的类型，例如 `LoraModel`
  - `peft_types` (`dict[str, str]`): 适配器名称到适配器类型的映射，例如 `{"default": "LORA"}`
  - `trainable_params` (`int`): 模型中可训练参数的数量
  - `total_params` (`int`): 模型中参数的总数
  - `num_adapter_layers` (`int`): 模型中适配器层的数量
  - `enabled` (`bool`, `Literal["irregular"]`): 是否所有适配器层都启用。如果有些启用有些不启用，这将是 `"irregular"`。这意味着您的模型处于不一致状态，可能无法按预期工作
  - `active_adapters` (`list[str]`, `Literal["irregular"]`): 活动适配器的名称。如果活动适配器在所有层中不一致，这将是 `"irregular"`，这意味着您的模型处于不一致状态，可能无法按预期工作
  - `merged_adapters` (`list[str]`, `Literal["irregular"]`): 合并适配器的名称。如果合并适配器在所有层中不一致，这将是 `"irregular"`，这意味着您的模型处于不一致状态，可能无法按预期工作
  - `available_adapters` (`list[str]`): 可用适配器的名称，例如 `["default"]`
- **传入参数**：无
- **返回参数**：`TunerModelStatus` - 包含模型状态的数据类

## 属性访问器方法

### `peft_config`
- **方法描述**：获取 PEFT 配置的属性访问器
- **传入参数**：无
- **返回参数**：`dict[str, PeftConfig]` - 适配器名称到配置对象的映射

### `active_adapters`
- **方法描述**：获取活动适配器列表的属性访问器
- **传入参数**：无
- **返回参数**：`list[str]` - 活动适配器名称列表

### `base_model_torch_dtype`
- **方法描述**：获取基础模型 torch 数据类型的属性访问器
- **传入参数**：无
- **返回参数**：基础模型的数据类型或 None

### `active_peft_config`
- **方法描述**：获取活动 PEFT 配置的属性访问器
- **传入参数**：无
- **返回参数**：当前活动适配器的 PEFT 配置

### `modules_to_save`
- **方法描述**：获取要保存模块的属性访问器
- **传入参数**：无
- **返回参数**：`Optional[set[str]]` - 要保存的模块名称集合，如果没有则为 None

## 工具和辅助方法

### `get_base_model`
- **方法描述**：返回基础模型
- **传入参数**：无
- **返回参数**：`torch.nn.Module` - 基础模型实例

### `disable_adapter`
- **方法描述**：禁用适配器模块的上下文管理器。使用它在基础模型上运行推理
- **传入参数**：无
- **返回参数**：上下文管理器

### `prepare_model_for_gradient_checkpointing`
- **方法描述**：在必要时为梯度检查点准备模型
- **传入参数**：
  - `model` (`PreTrainedModel`): 要准备的模型
- **返回参数**：准备好的模型

### `create_or_update_model_card`
- **方法描述**：更新或创建模型卡片以包含关于 peft 的信息：
  1. 添加 `peft` 库标签
  2. 添加 peft 版本
  3. 添加基础模型信息
  4. 如果使用，添加量化信息
- **传入参数**：
  - `output_dir` (`str`): 输出目录
- **返回参数**：无

## 提示学习相关方法

### `_setup_prompt_encoder`
- **方法描述**：设置提示编码器（仅适用于提示学习方法）
- **传入参数**：
  - `adapter_name` (`str`): 适配器名称
- **返回参数**：无

#### method 解读
```python
# 获取指定适配器的配置
config = self.peft_config[adapter_name]

# 如果提示编码器模块不存在，则初始化提示编码器和提示令牌字典
if not hasattr(self, "prompt_encoder"):
    self.prompt_encoder = torch.nn.ModuleDict({})  # 存储不同适配器的提示编码器
    self.prompt_tokens = {}  # 存储不同适配器的提示令牌

# 初始化 transformer 主干模型
transformer_backbone = None
# 遍历基础模型的直接子模块
for name, module in self.base_model.named_children():
    # 冻结所有子模块的参数，只训练提示相关参数
    for param in module.parameters():
        param.requires_grad = False
    # 如果是 PreTrainedModel 实例，将其标记为 transformer 主干
    if isinstance(module, PreTrainedModel):
        # Make sure to freeze Tranformers model
        if transformer_backbone is None:
            transformer_backbone = module
            self.transformer_backbone_name = name  # 保存主干模块名称

# 如果没有找到 transformer 主干，则使用整个基础模型
if transformer_backbone is None:
    transformer_backbone = self.base_model

# 如果没有指定 transformer 子模块数量，则根据任务类型设置
if config.num_transformer_submodules is None:
    # SEQ_2_SEQ_LM 任务（如 T5）需要 2 个子模块（编码器和解码器）
    # 其他任务只需要 1 个子模块
    config.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1

# 确定词嵌入层的位置
word_embeddings = None
try:
    # 首先尝试通过标准路径找到词嵌入（适用于 BERT、RoBERTa、DeBERTa 等模型）
    word_embeddings = self.base_model.get_submodule("embeddings.word_embeddings")
except AttributeError:
    pass

# 如果通过标准路径没有找到词嵌入，则通过参数大小推断
if word_embeddings is None:
    # 遍历 transformer 主干的所有命名参数，找到与词汇表大小匹配的参数
    for named_param, value in list(transformer_backbone.named_parameters()):
        # 处理 ZeRO-3 分布式训练情况，DeepSpeed 会将分片张量修改为形状 [0]
        # 实际的未分片形状存储在 "ds_shape" 属性中
        deepspeed_distributed_tensor_shape = getattr(value, "ds_shape", None)

        # 处理多模态模型（VLM）的情况，获取文本配置中的词汇表大小
        if hasattr(self.base_model.config, "get_text_config"):
            vocab_size = self.base_model.config.get_text_config().vocab_size
        # 兼容旧版本 transformers 的多模态配置
        elif "text_config" in self.base_model.config:
            vocab_size = self.base_model.config.text_config.vocab_size
        else:
            vocab_size = self.base_model.config.vocab_size

        # 检查参数的第一个维度是否等于词汇表大小（词嵌入矩阵的特征）
        if value.shape[0] == vocab_size or (
            deepspeed_distributed_tensor_shape is not None
            and deepspeed_distributed_tensor_shape[0] == vocab_size
        ):
            # 获取该参数对应的模块（去掉 ".weight" 后缀）
            word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
            break

# 保存找到的词嵌入模块
self.word_embeddings = word_embeddings

# 根据 PEFT 类型获取对应的调优器类
model_cls = PEFT_TYPE_TO_TUNER_MAPPING[config.peft_type]

# 根据不同的提示学习类型创建相应的提示编码器
if config.peft_type in (PeftType.PROMPT_TUNING, PeftType.MULTITASK_PROMPT_TUNING, PeftType.CPT):
    # 提示调优、多任务提示调优、CPT：需要词嵌入信息
    prompt_encoder = model_cls(config, self.word_embeddings)
elif config.peft_type == PeftType.P_TUNING:
    # P-Tuning：只需要配置信息
    prompt_encoder = model_cls(config)
elif config.peft_type == PeftType.PREFIX_TUNING:
    # 前缀调优：需要检查是否与梯度检查点兼容
    # prefix tuning 现在使用 Cache，但与梯度检查点不兼容
    if any(getattr(module, "gradient_checkpointing", False) for module in self.get_base_model().modules()):
        raise ValueError("Prefix tuning does not work with gradient checkpointing.")
    prompt_encoder = model_cls(config)
else:
    # 不支持的提示学习类型
    raise ValueError("Not supported")

# 将提示编码器移动到正确的设备上
prompt_encoder = prompt_encoder.to(self.device)

# 将新创建的提示编码器添加到 ModuleDict 中
self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder}))

# 为适配器创建提示令牌张量
# 范围：0 到 (虚拟令牌数 * transformer 子模块数 - 1)
self.prompt_tokens[adapter_name] = torch.arange(
    config.num_virtual_tokens * config.num_transformer_submodules
).long()
```

### `get_prompt_embedding_to_save`
- **方法描述**：返回保存模型时要保存的提示嵌入。仅在使用提示学习方法时适用
- **传入参数**：
  - `adapter_name` (`str`): 适配器名称
- **返回参数**：`torch.Tensor` - 提示嵌入张量

### `get_prompt`
- **方法描述**：返回用于 Peft 的虚拟提示。仅在使用提示学习方法时适用
- **传入参数**：
  - `batch_size` (`int`): 批次大小
  - `task_ids` (`torch.Tensor`, *可选*): 任务 ID
  - `max_cache_len` (`int`, *可选*): 最大缓存长度
- **返回参数**：`torch.Tensor` - 虚拟提示张量

#### method 解读
```python
# 获取当前活动适配器的配置和提示编码器
peft_config = self.active_peft_config
prompt_encoder = self.prompt_encoder[self.active_adapter]

# 准备提示令牌张量：扩展到指定的批次大小并移动到正确的设备
prompt_tokens = (
    self.prompt_tokens[self.active_adapter]
    .unsqueeze(0)  # 添加批次维度
    .expand(batch_size, -1)  # 扩展到指定的批次大小
    .to(prompt_encoder.embedding.weight.device)  # 移动到编码器权重所在的设备
)

# 根据不同的提示学习类型生成提示
if peft_config.peft_type == PeftType.PREFIX_TUNING:
    # 前缀调优：生成 past_key_values 用于注意力机制
    # 只使用前 n 个虚拟令牌
    prompt_tokens = prompt_tokens[:, : peft_config.num_virtual_tokens]

    if peft_config.inference_mode:
        # 推理模式：直接使用编码器权重，不进行前向传播
        past_key_values = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
    else:
        # 训练模式：通过编码器前向传播生成提示
        past_key_values = prompt_encoder(prompt_tokens)

    # 转换数据类型以匹配基础模型
    if self.base_model_torch_dtype is not None:
        past_key_values = past_key_values.to(self.base_model_torch_dtype)

    # 重塑张量以适应注意力机制的结构
    # [batch_size, num_virtual_tokens, num_layers*2, num_heads, head_dim]
    past_key_values = past_key_values.view(
        batch_size,
        peft_config.num_virtual_tokens,
        peft_config.num_layers * 2,  # *2 因为有 key 和 value
        peft_config.num_attention_heads,
        peft_config.token_dim // peft_config.num_attention_heads,  # head_dim
    )

    # 对于编码器-解码器模型，复制一份用于解码器
    if peft_config.num_transformer_submodules == 2:
        past_key_values = torch.cat([past_key_values, past_key_values], dim=2)

    # 重新排列维度：[num_layers*2, batch_size, num_heads, num_virtual_tokens, head_dim]
    # 然后分割成编码器和解码器的缓存
    past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
        peft_config.num_transformer_submodules * 2
    )

    # 获取基础模型配置以进行后处理
    base_model = self.get_base_model()
    model_config = getattr(base_model, "config", None)
    model_type = getattr(model_config, "model_type", "")

    # 根据模型类型应用特定的后处理函数
    if TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING.get(self.config.model_type, None) is not None:
        # 使用模型特定的后处理函数
        post_process_fn = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[self.config.model_type]
        past_key_values = post_process_fn(past_key_values)
    elif ("gemma2" in model_type) or ("gemma3_text" in model_type):
        # Gemma2 和 Gemma3 特殊处理：使用 HybridCache
        if max_cache_len is None:
            raise ValueError(
                "max_cache_len is None but it should have been passed. Something went wrong, please open an "
                "issue on GitHub with a reproducer: https://github.com/huggingface/peft/issues"
            )
        base_config = base_model.config
        if hasattr(base_config, "get_text_config"):
            base_config = base_config.get_text_config()

        # 创建 HybridCache 实例
        new_cache = HybridCache(
            base_config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            dtype=past_key_values[0].dtype,
            device=past_key_values[0].device,
        )

        # 更新缓存中的键值对
        cache_position = torch.arange(peft_config.num_virtual_tokens, device=past_key_values[0].device)
        for layer_idx in range(peft_config.num_layers):
            key_states, value_states = past_key_values[0][layer_idx], past_key_values[1][layer_idx]
            new_cache.update(
                key_states, value_states, layer_idx, cache_kwargs={"cache_position": cache_position}
            )
        past_key_values = new_cache
    elif peft_config.num_transformer_submodules == 1:
        # 单模块模型：使用 DynamicCache
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
    elif (peft_config.num_transformer_submodules == 2) and getattr(
        self.base_model, "_supports_cache_class", True
    ):
        # 编码器-解码器模型：使用 EncoderDecoderCache
        past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
        past_key_values.cross_attention_cache = DynamicCache()
        past_key_values.is_updated = {
            layer_idx: False for layer_idx in range(len(past_key_values.cross_attention_cache.key_cache))
        }

    # 确保缓存张量在正确的设备上
    map_cache_to_layer_device_map(self.get_base_model(), past_key_values)
    return past_key_values
else:
    # 其他提示学习方法（Prompt Tuning, P-Tuning, Multitask Prompt Tuning 等）
    if peft_config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
        # 多任务提示调优：需要任务 ID
        prompts = prompt_encoder(prompt_tokens, task_ids)
    else:
        # 单任务提示调优
        if peft_config.inference_mode:
            # 推理模式：直接使用编码器权重
            prompts = prompt_encoder.embedding.weight
        else:
            # 训练模式：优化策略 - 只处理一个样本然后重复输出
            # 这是为了提高效率，避免重复计算相同的编码结果
            # 参考: https://github.com/huggingface/peft/issues/2043#issuecomment-2321522577
            prompt_tokens = prompt_tokens[:1]  # 只使用第一个样本的令牌
            prompts = prompt_encoder(prompt_tokens)  # 编码一次

        # 重复编码结果以匹配批次大小
        prompts = prompts.repeat(batch_size, 1, 1)
    return prompts
```

## 内部和特殊方法

### `__getattr__`
- **方法描述**：将缺失属性转发到包装模块
- **传入参数**：
  - `name` (`str`): 属性名称
- **返回参数**：属性值

### `_enable_peft_forward_hooks`
- **方法描述**：启用 PEFT 前向钩子的上下文管理器
- **传入参数**：
  - `*args`: 位置参数
  - `**kwargs`: 关键字参数
- **返回参数**：上下文管理器

### `_split_kwargs`
- **方法描述**：分割 kwargs 的类方法
- **传入参数**：
  - `kwargs` (`dict[str, Any]`): 要分割的 kwargs
- **返回参数**：`tuple` - (hf_hub_download_kwargs, other_kwargs)

### `_update_offload`
- **方法描述**：更新磁盘卸载模块的 offload_index 和 safetensors 文件，用于加载和合并 PeftModels
- **传入参数**：
  - `offload_index` (`dict[str, dict[str, str]]`): 磁盘卸载模块的字典，包含其元数据和 safetensors 文件名
  - `adapters_weights` (`dict[str, torch.tensor]`): Peft 适配器模块名称和权重的字典
- **返回参数**：更新后的 offload_index

### `_check_new_adapter_config`
- **方法描述**：对新添加的 PEFT 配置执行检查以确保完整性
- **传入参数**：
  - `peft_config` (`PeftConfig`): 要检查的 PEFT 配置
  - `is_trainable` (`bool`): 是否可训练
- **返回参数**：无