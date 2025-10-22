# PeftModel 概述

PeftModel 是 PEFT（Parameter-Efficient Fine-Tuning）库的核心模型类，它封装了各种参数高效微调方法，为不同的预训练模型提供统一的接口。PeftModel 继承自 `PushToHubMixin` 和 `torch.nn.Module`，支持多种微调技术，如 LoRA、Prefix Tuning、Prompt Tuning 等。

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

### `add_adapter`
- **方法描述**：根据传入的配置向模型添加适配器。此适配器未经过训练。要加载训练好的适配器，请使用 [`PeftModel.load_adapter`]。新适配器的名称应该是唯一的。新适配器不会自动设置为活动适配器。
- **传入参数**：
  - `adapter_name` (`str`): 要添加的适配器名称
  - `peft_config` (`PeftConfig`): 要添加的适配器配置
  - `low_cpu_mem_usage` (`bool`, *可选*, 默认 `False`): 在 meta 设备上创建空的适配器权重。用于加速加载保存适配器的过程。创建新的 PEFT 适配器进行训练时不要使用此选项
- **返回参数**：无

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