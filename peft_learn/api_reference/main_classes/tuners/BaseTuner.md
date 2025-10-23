# BaseTuner 概述

`BaseTuner` 是一个基础调优器模型，为所有可注入到 torch.nn.Module 的调优器提供通用方法和属性。

## 类的描述

`BaseTuner` 是 PEFT (Parameter-Efficient Fine-Tuning) 库中所有调优器的抽象基类。它定义了适配器注入和管理的通用框架，为不同的参数高效微调方法（如 LoRA、AdaLoRA 等）提供统一的接口和基础功能。

核心方法：
- [\_\_init\_\_](#__init__)
  - [inject_adapter](#inject_adapter)
    - [_check_target_module_exists](../../adapters/lora/LoraModel.md#_check_target_module_existsstatic)
      - [check_target_module_exists](utils.md#check_target_module_exists)
    - [_create_and_replace](../../adapters/lora/LoraModel.md#_create_and_replace)
    - [_mark_only_adapters_as_trainable](../../adapters/lora/LoraModel.md#_mark_only_adapters_as_trainable)

## 需要重写的方法

### 抽象方法（必须在子类中实现）

- **_prepare_adapter_config**:
  - 描述：一个私有方法，用于准备适配器配置。例如，当 `target_modules` 字段缺失时，可以自动推断目标模块。
  - 参考：`peft.tuners.lora.LoraModel._prepare_adapter_config` 中的实现示例

- **_create_and_replace**:
  - 描述：私有方法，用于创建适配器模块并用其替换目标模块。这是适配器注入的核心方法。
  - 参考：`peft.tuners.lora.LoraModel._create_and_replace` 中的实现示例

- **_check_target_module_exists**:
  - 描述：私有辅助方法，检查传递的模块键名是否与 `adapter_config.target_modules` 列表中的任何目标模块匹配。
  - 返回：如果匹配返回 `True`，否则返回 `False`

- **_mark_only_adapters_as_trainable**:
  - 描述：辅助方法，仅标记适配器层为可训练（即设置 `module.requires_grad = False`）。需要重写以匹配正确的键名。
  - 参考：`peft.tuners.lora.LoraModel._mark_only_adapters_as_trainable` 中的实现示例

- **disable_adapter_layers**:
  - 描述：就地禁用所有适配器层。

- **enable_adapter_layers**:
  - 描述：就地启用所有适配器层。

## 类属性

- **model** (`torch.nn.Module`):
  - 描述：要附加适配器调优器层的模型。

- **forward** (`Callable`):
  - 描述：模型的前向传播方法。

- **peft_config** (`Union[PeftConfig, dict[str, PeftConfig]]`):
  - 描述：适配器配置对象，应该是 `str` 到 `PeftConfig` 对象的字典。也可以传递 `PeftConfig` 对象，将创建默认名称为 `adapter` 的新适配器。

- **config** (`dict[str, Any]`):
  - 描述：模型配置对象，应该是 `str` 到 `Any` 对象的字典。

- **targeted_module_names** (`list[str]`):
  - 描述：实际被适配的模块名称列表。可用于检查 `config.target_modules` 是否正确指定。

- **targeted_parameter_names** (`list[str]`):
  - 描述：实际被适配的参数名称列表。可用于检查 `config.target_parameters` 是否正确指定。

# method 概述

## 核心初始化和基础方法

**核心方法** ：
- [inject_adapter](#inject_adapter)

### \_\_init\_\_
- 描述：初始化 BaseTuner 实例，设置模型、配置和相关属性。
- 输入参数:
  - model (`torch.nn.Module`): 要调优的模型
  - peft_config (`Union[PeftConfig, dict[str, PeftConfig]]`): 适配器配置
  - adapter_name (`str`): 适配器名称
  - low_cpu_mem_usage (`bool`, 默认值: `False`): 是否使用低 CPU 内存模式
  - state_dict (`Optional[dict[str, torch.Tensor]]`, 默认值: `None`): 可选的状态字典
- 输出参数: `None`

```python
# 调用父类初始化
super().__init__()

# 设置基础属性
self.model = model
self.targeted_module_names: list[str] = []       # 存储实际被适配的模块名称
self.targeted_parameter_names: list[str] = []    # 存储实际被适配的参数名称

# 处理peft_config的多种情况
if not hasattr(self, "peft_config"):
    # 首次初始化，创建peft_config字典
    self.peft_config = {adapter_name: peft_config} if isinstance(peft_config, PeftConfig) else peft_config
else:
    # 模型已存在peft_config，处理多适配器情况
    warnings.warn(
        "Already found a `peft_config` attribute in the model. This will lead to having multiple adapters"
        " in the model. Make sure to know what you are doing!"
    )
    if isinstance(peft_config, PeftConfig):
        # 添加单个适配器配置
        self.peft_config[adapter_name] = peft_config
    else:
        # 合并适配器配置字典
        self.peft_config.update(peft_config)

# 设置活跃适配器
self.active_adapter: str | list[str] = adapter_name

# 执行预注入钩子（子类可重写）
self._pre_injection_hook(self.model, self.peft_config[adapter_name], adapter_name)

# 条件性注入适配器（排除XLora类型）
if peft_config != PeftType.XLORA or peft_config[adapter_name] != PeftType.XLORA:
    self.inject_adapter(self.model, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage, state_dict=state_dict)

# 将配置复制到注入的模型中
self.model.peft_config = self.peft_config
```

### forward
- 描述：模型的前向传播方法，直接调用底层模型的 forward 方法。
- 输入参数:
  - *args (`Any`): 可变位置参数
  - **kwargs (`Any`): 可变关键字参数
- 输出参数: `Any` - 模型前向传播的输出

### active_adapters (property)
- 描述：返回当前活跃适配器的列表。
- 输入参数: 无
- 输出参数: `list[str]` - 活跃适配器名称列表

## 适配器配置和准备方法

### _pre_injection_hook
- 描述：在适配器注入到模型之前调用的钩子方法。子类可以重写此方法以执行任何预注入操作。
- 输入参数:
  - model (`nn.Module`): 要适配的模型
  - config (`PeftConfig`): 适配器配置
  - adapter_name (`str`): 适配器名称
- 输出参数: `None`

### _prepare_adapter_config (abstract)
- 描述：准备适配器配置的私有抽象方法。如果 `peft_config.target_modules` 为 `None`，可以从 `TRANSFORMERS_MODELS_TO_XXX_TARGET_MODULES_MAPPING` 自动推断目标模块。
- 输入参数:
  - peft_config (`PeftConfig`): 适配器配置
  - model_config (`dict`): transformers 模型配置，应包含 `model_type` 键
- 输出参数: `PeftConfig` - 准备好的适配器配置

### _prepare_model
- 描述：在应用适配器之前修改模型结构的私有方法。
- 输入参数:
  - peft_config (`PeftConfig`): 准备好的适配器配置
  - model (`nn.Module`): 将要适配的模型
- 输出参数: `None`

### _check_new_adapter_config
- 描述：添加新适配器时检查配置的辅助方法。如果配置有问题或与现有适配器冲突，抛出 ValueError。
- 输入参数:
  - config (`PeftConfig`): 要检查的配置
- 输出参数: `None`

## 模块检测和替换方法

### _check_target_module_exists(abstract)
- 描述：检查传递的模块键名是否与 `peft_config.target_modules` 列表中的任何目标模块匹配的私有辅助方法。
- 输入参数:
  - peft_config (`PeftConfig`): 适配器配置
  - key (`str`): 模块的键名
- 输出参数: `bool` - 如果匹配返回 `True`，否则返回 `False`

### _create_and_replace(abstract)
- 描述：用适配器层就地替换目标模块的抽象方法。所有调优器类都需要重写此方法。
- 输入参数:
  - peft_config (`PeftConfig`): 适配器配置
  - adapter_name (`str`): 适配器名称
  - target (`nn.Module`): 目标模块
  - target_name (`str`): 目标模块名称
  - parent (`nn.Module`): 父模块
  - current_key (`str`): 当前适配目标的键
  - parameter_name (`Optional[str]`, 默认值: `None`): 如果目标是 `nn.Parameter`，则为参数名称
- 输出参数: `None`

### _create_and_replace_parameter
- 描述：创建和替换参数的方法，如果不支持目标 `nn.Parameter` 则抛出 NotImplementedError。
- 输入参数:
  - peft_config: 适配器配置
  - adapter_name (`str`): 适配器名称
  - target: 目标模块
  - target_name (`str`): 目标名称
  - parent: 父模块
  - current_key: 当前键
- 输出参数: `None`

### _check_target_module_compatiblity
- 描述：防止在特定架构（如 Mamba）中将 LoRA 应用于不兼容模块的方法。
- 输入参数:
  - peft_config (`PeftConfig`): 适配器配置
  - model (`nn.Module`): 模型
  - target_name (`str`): 目标名称
- 输出参数: `None`

## 适配器注入和管理方法

### inject_adapter
- 描述：创建适配器层并用适配器层替换目标模块。如果传递非提示调优适配器类，此方法由 `peft.mapping.get_peft_model` 在底层调用。
- 输入参数:
  - model (`nn.Module`): 要调优的模型
  - adapter_name (`str`): 适配器名称
  - autocast_adapter_dtype (`bool`, 默认值: `True`): 是否自动转换适配器数据类型
  - low_cpu_mem_usage (`bool`, 默认值: `False`): 在 meta 设备上创建空适配器权重，用于加速加载过程
  - state_dict (`Optional[dict[str, torch.Tensor]]`, 默认值: `None`): 如果传递状态字典，适配器将基于状态字典条目注入
- 输出参数: `None`

#### PREPARATION OF MODEL AND CONFIG
这部分代码负责准备模型和配置，进行各种检查和优化设置。

```python
# 获取适配器配置并初始化追踪列表
peft_config = self.peft_config[adapter_name]
excluded_modules = []                    # 被排除的模块列表
unmatched_modules = []                   # 未匹配的模块列表
targeted_modules_from_peft_config: list[str] = []  # 仅在使用state_dict时相关

# 在方法开始时执行所有检查，避免模型处于半初始化状态
self._check_new_adapter_config(peft_config)

# 获取并准备模型配置
model_config = self.get_model_config(model)
peft_config = self._prepare_adapter_config(peft_config, model_config)
self._prepare_model(peft_config, model)

# 检查target_parameters与state_dict的兼容性
if getattr(peft_config, "target_parameters", []) and state_dict:
    raise ValueError(
        "Trying to inject a PEFT adapter from a state_dict but the PEFT config uses `target_parameters`. This "
        "is not supported -- when using `target_parameters`, please inject the adapter without the state_dict."
    )

# 获取模型的命名模块列表
named_modules = list(model.named_modules())
key_list = [key for key, _ in named_modules]

# 处理虚拟目标模块的情况
uses_dummy_target_modules = getattr(peft_config, "target_modules", None) == DUMMY_TARGET_MODULES
if uses_dummy_target_modules:
    # 虚拟适配器，允许不匹配任何模块
    named_modules = []
    key_list = []

# 更新peft_config.target_modules（如将"all-linear"转换为具体线性层）
peft_config = _maybe_include_all_linear_layers(peft_config, model)

# 优化target_modules列表以减少匹配开销
# 当target_modules很大时（如从diffusers加载的LoRA），需要进行优化
if (
    isinstance(peft_config.target_modules, (list, set))
    and (len(peft_config.target_modules) >= MIN_TARGET_MODULES_FOR_OPTIMIZATION)
    and (peft_config.peft_type != PeftType.IA3)  # IA³因为有feedforward_modules耦合，跳过优化
):
    # 找出不匹配target_modules的模块名
    names_no_target = [
        name
        for name in key_list
        if not any((name == suffix) or name.endswith("." + suffix) for suffix in peft_config.target_modules)
    ]
    # 计算最小的target_modules集合
    new_target_modules = _find_minimal_target_modules(peft_config.target_modules, names_no_target)
    if len(new_target_modules) < len(peft_config.target_modules):
        peft_config.target_modules = new_target_modules
```

#### MATCHING & CREATING MODULES
这部分代码负责匹配目标模块并创建适配器层。(**关键代码**)

```python
# 构建已存在的适配器映射，避免干扰现有适配器
existing_adapter_map = {}
for key, module in named_modules:
    if isinstance(module, BaseTunerLayer):
        existing_adapter_map[key] = module

# 从state_dict中提取模块名（如果提供）
module_names: set[str] = set()
if state_dict is not None:
    prefix = PEFT_TYPE_TO_PREFIX_MAPPING[peft_config.peft_type]
    # 从state_dict的键中移除适配器前缀，得到模块名
    module_names = {k.rsplit("." + prefix, 1)[0] for k in state_dict}

# 遍历所有模块进行匹配和替换
for key, module in named_modules:
    if not key:
        continue

    # 检查是否属于已存在的适配器，避免干扰
    for adapter_key in existing_adapter_map:
        if key.startswith(adapter_key + "."):
            excluded_modules.append(key)
            break

    if excluded_modules and excluded_modules[-1] == key:
        continue

    if state_dict is None:
        # 正常机制：使用peft_config匹配模块
        result = self._check_target_module_exists(peft_config, key)
        if isinstance(result, _ExcludedModule):
            excluded_modules.append(key)
        elif not result:
            unmatched_modules.append(key)
        else:
            # 匹配成功，创建适配器层
            self.targeted_module_names.append(key)  # [batch_size, seq_len, hidden_dim] 目标模块
            parent, target, target_name = _get_submodules(model, key)
            self._check_target_module_compatiblity(peft_config, model, target_name)
            ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
            with ctx():
                self._create_and_replace(
                    peft_config, adapter_name, target, target_name, parent, current_key=key
                )
    else:
        # 使用state_dict匹配模块
        if key not in module_names:
            unmatched_modules.append(key)
        else:
            # 基于state_dict创建适配器层
            self.targeted_module_names.append(key)
            parent, target, target_name = _get_submodules(model, key)
            self._check_target_module_compatiblity(peft_config, model, target_name)
            ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
            with ctx():
                self._create_and_replace(
                    peft_config, adapter_name, target, target_name, parent, current_key=key
                )

        # 记录通过配置会匹配的模块，用于后续比较
        if self._check_target_module_exists(peft_config, key):
            targeted_modules_from_peft_config.append(key)

# 处理target_parameters配置
if getattr(peft_config, "target_parameters", []):
    self._inject_parameters(
        peft_config=peft_config, model=model, adapter_name=adapter_name, low_cpu_mem_usage=low_cpu_mem_usage
    )
```

#### CHECK FOR ERRORS
这部分代码进行错误检查和警告处理。

```python
# 检查state_dict与PEFT配置的一致性
if state_dict is not None:
    targeted_set_from_peft_config = set(targeted_modules_from_peft_config)
    targeted_set_from_state_dict = set(self.targeted_module_names)
    diff_peft_config = targeted_set_from_peft_config - targeted_set_from_state_dict
    diff_state_dict = targeted_set_from_state_dict - targeted_set_from_peft_config
    warning_msg = ""
    if diff_peft_config or diff_state_dict:
        warning_msg = (
            "While injecting the PEFT adapters, an inconsistency was discovered between the PEFT config and "
            "the provided state_dict. This is not necessarily an issue and can be ignored if this was the "
            "intent. "
        )
    if diff_peft_config:
        warning_msg += (
            f"The PEFT config contained these additional target modules: {sorted(diff_peft_config)}. "
        )
    if diff_state_dict:
        warning_msg += f"The state_dict contained these additional target modules: {sorted(diff_state_dict)}. "
    if warning_msg:
        warnings.warn(warning_msg, RuntimeWarning)

# 检查是否有模块被适配（排除虚拟适配器情况）
if not self.targeted_module_names and not self.targeted_parameter_names and not uses_dummy_target_modules:
    if excluded_modules and not unmatched_modules:
        # 所有目标模块都被排除
        raise ValueError(
            "All modules were excluded. This is likely unintended. "
            "Check your `target_modules`, `exclude_modules` and `modules_to_save` configuration."
        )
    elif not excluded_modules and unmatched_modules and not peft_config.target_modules:
        # 没有传递target_modules且没有找到target_parameters
        raise ValueError(
            "No `target_modules` passed but also no `target_parameters` found. Please check the values for "
            "these arguments."
        )
    elif not excluded_modules and unmatched_modules:
        # 没有目标模块匹配
        error_msg = (
            f"Target modules {peft_config.target_modules} not found in the base model. "
            f"Please check the target modules and try again."
        )
        # 添加layers_to_transform相关信息
        if getattr(peft_config, "layers_to_transform", None) is not None:
            error_msg += f" Note: You specified 'layers_to_transform': {peft_config.layers_to_transform}."
        if getattr(peft_config, "layers_pattern", None) is not None:
            error_msg += f" You also specified 'layers_pattern': {peft_config.layers_pattern}."
        raise ValueError(error_msg)
    else:
        # 部分模块不匹配，部分被排除
        error_msg = (
            "No modules were targeted for adaptation. "
            "This might be caused by a combination of mismatched target modules and excluded modules. "
            "Please check your `target_modules` and `exclude_modules` configuration. You may also have "
            "only targeted modules that are marked to be saved (`modules_to_save`)."
        )
        if getattr(peft_config, "layers_to_transform", None) is not None:
            error_msg += f" Note: You specified 'layers_to_transform': {peft_config.layers_to_transform}."
        if getattr(peft_config, "layers_pattern", None) is not None:
            error_msg += f" You also specified 'layers_pattern': {peft_config.layers_pattern}."
        raise ValueError(error_msg)

# 检查exclude_modules是否被使用
elif hasattr(peft_config, "exclude_modules") and peft_config.exclude_modules and not excluded_modules:
    warnings.warn(
        f"You have passed exclude_modules={peft_config.exclude_modules} but no modules were excluded. "
        "Please check that exclude_modules was set correctly."
    )

# 检查target_modules和target_parameters的匹配情况
elif not uses_dummy_target_modules:
    if peft_config.target_modules and not self.targeted_module_names:
        warnings.warn(
            f"target_modules={peft_config.target_modules} were set but no module was matched.", RuntimeWarning
        )
    elif getattr(peft_config, "target_parameters", []) and not self.targeted_parameter_names:
        warnings.warn(
            f"target_parameters={peft_config.target_parameters} were set but no parameter was matched.",
            RuntimeWarning,
        )

# 检查绑定的目标模块警告
tied_target_modules = self._get_tied_target_modules(model=model)
if tied_target_modules:
    warnings.warn(
        f"Model with `tie_word_embeddings=True` and the {tied_target_modules=} are part of the adapter. "
        "This can lead to complications, for example when merging the adapter "
        "or converting your model to formats other than safetensors. "
        "See for example https://github.com/huggingface/peft/issues/2018."
    )
```

#### HOUSEKEEPING
这部分代码进行最终的设置和清理工作。

```python
# 重新设置活跃适配器，确保正确的适配器被激活
self.set_adapter(self.active_adapters)
# 标记只有适配器层为可训练
self._mark_only_adapters_as_trainable(model)

# 如果是推理模式，禁用适配器参数的梯度
if self.peft_config[adapter_name].inference_mode:
    for n, p in model.named_parameters():
        if adapter_name in n:
            p.requires_grad = False

# 设置额外的可训练模块
set_additional_trainable_modules(
    model=model,
    peft_config=peft_config,
    model_config=BaseTuner.get_model_config(self),
    adapter_name=adapter_name,
)
```

### _inject_parameters
- 描述：注入参数的方法，用于处理 `target_parameters` 配置。
- 输入参数:
  - peft_config (`PeftConfig`): 适配器配置
  - model (`nn.Module`): 模型
  - adapter_name (`str`): 适配器名称
  - low_cpu_mem_usage (`bool`): 是否使用低 CPU 内存模式
- 输出参数: `None`

### _delete_auxiliary_adapter
- 描述：删除辅助适配器的方法。
- 输入参数:
  - adapter_name (`str`): 要删除的适配器名称
  - new_active_adapters (`Optional[list[str]]`): 新的活跃适配器列表
- 输出参数: `None`

### _unloading_checks
- 描述：卸载适配器前的检查方法。
- 输入参数:
  - adapter_names (`Optional[list[str]]`): 要考虑的适配器名称
- 输出参数: `None`

## 适配器训练和启用控制方法

### _mark_only_adapters_as_trainable (abstract)
- 描述：仅标记适配器层为可训练（设置 `module.requires_grad = False`）的辅助抽象方法。
- 输入参数:
  - model (`nn.Module`): 要标记的模型
- 输出参数: `None`

### disable_adapter_layers (abstract)
- 描述：就地禁用所有适配器。
- 输入参数: 无
- 输出参数: `None`

### enable_adapter_layers (abstract)
- 描述：就地启用所有适配器。
- 输入参数: 无
- 输出参数: `None`

## 适配器合并和取消合并方法

### merge_adapter
- 描述：将适配器层合并到基础模型中。合并适配器可以加速前向传播。
- 输入参数:
  - adapter_names (`Optional[list[str]]`, 默认值: `None`): 应合并的适配器名称列表，如果为 `None` 则合并所有活跃适配器
  - safe_merge (`bool`, 默认值: `False`): 如果为 `True`，合并操作将在原始权重的副本中执行并在合并前检查 NaN
- 输出参数: `None`

### unmerge_adapter
- 描述：从基础模型中取消合并所有已合并的适配器层。
- 输入参数: 无
- 输出参数: `None`

### _check_merge_allowed
- 描述：检查是否可以合并适配器的辅助方法。如果无法合并给定配置的适配器，抛出 ValueError。
- 输入参数: 无
- 输出参数: `None`

## 数据类型和设备管理方法

### _cast_adapter_dtype
- 描述：将适配器权重转换为正确数据类型的辅助方法。目前只将 float16 和 bfloat16 上转换为 float32。
- 输入参数:
  - adapter_name (`str`): 适配器名称
  - autocast_adapter_dtype (`bool`, 默认值: `True`): 是否自动转换适配器数据类型
- 输出参数: `None`

## 模型配置和查询方法

### get_model_config (static)
- 描述：以字典形式从模型获取配置的方法。如果模型没有 config 属性，则返回默认配置。
- 输入参数:
  - model (`nn.Module`): 要获取配置的模型
- 输出参数: `dict` - 模型配置字典

### _get_tied_target_modules
- 描述：获取绑定目标模块列表的方法。
- 输入参数:
  - model (`nn.Module`): 模型
- 输出参数: `list[str]` - 绑定的目标模块列表