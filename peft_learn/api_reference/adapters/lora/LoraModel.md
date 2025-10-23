# LoraModel

## 概述

`LoraModel` 从预训练的 transformers 模型创建低秩适配器（LoRA）模型。该方法的详细描述见论文：https://huggingface.co/papers/2106.09685。

LoRA 通过在预训练模型的权重矩阵旁添加低秩矩阵来实现高效的参数微调，大幅减少可训练参数数量，同时保持模型性能。

**核心方法** ：
- [LoraModel._create_and_replace](#_create_and_replace)
  - [LoraModel._create_new_module (static)](#_create_new_modulestatic)
    - [LoraLayer.dispatch_default](LoraLayer.md#dispatch_default)
      - [Linear.\_\_init\_\_](Linear.md#__init__)
  - [LoraModel._replace_module](#_replace_module)


## 参数

- **model** (`torch.nn.Module`): 要适配的模型
- **config** (`LoraConfig`): LoRA 模型的配置
- **adapter_name** (`str`): 适配器名称，默认为 `"default"`
- **low_cpu_mem_usage** (`bool`, 可选, 默认为 `False`): 在 meta 设备上创建空的适配器权重，用于加速加载过程

## 返回值

`torch.nn.Module`: LoRA 模型

## 使用案例

```python
from transformers import AutoModelForSeq2SeqLM
from peft import LoraModel, LoraConfig

config = LoraConfig(
    task_type="SEQ_2_SEQ_LM",
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.01,
)

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
lora_model = LoraModel(model, config, "default")
```

```python
import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

rank = ...
target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
config = LoraConfig(
    r=4, lora_alpha=16, target_modules=target_modules, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
)
quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "kakaobrain/kogpt",
    revision="KoGPT6B-ryan1.5b-float16",
    bos_token="[BOS]",
    eos_token="[EOS]",
    unk_token="[UNK]",
    pad_token="[PAD]",
    mask_token="[MASK]",
)
model = transformers.GPTJForCausalLM.from_pretrained(
    "kakaobrain/kogpt",
    revision="KoGPT6B-ryan1.5b-float16",
    pad_token_id=tokenizer.eos_token_id,
    use_cache=False,
    device_map={"": rank},
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
)
model = prepare_model_for_kbit_training(model)
lora_model = get_peft_model(model, config)
```

## 属性

- **model** (`~transformers.PreTrainedModel`): 要适配的模型
- **peft_config** (`LoraConfig`): LoRA 模型的配置


# method

## 配置和准备方法

### _check_new_adapter_config
- **方法描述**: 添加新适配器时检查配置的辅助方法。如果配置有问题或与现有适配器冲突，抛出 ValueError。
- **参数**:
  - config (`LoraConfig`): 要检查的配置
- **返回值**: `None`
- **使用案例**: 在添加多个适配器时，检查偏置设置的兼容性

### _check_target_module_exists(static)
- **方法描述**: 检查目标模块是否存在的静态方法
- **参数**:
  - lora_config: LoRA 配置
  - key: 模块键名
- **返回值**: `bool` - 模块是否存在
- **使用案例**: 在适配器注入过程中验证目标模块

### _prepare_model
- **方法描述**: 在应用适配器之前修改模型结构的私有方法
- **参数**:
  - peft_config (`PeftConfig`): 准备好的适配器配置
  - model (`nn.Module`): 将要适配的模型
- **返回值**: `None`
- **使用案例**: 根据配置中的 layer_replication 参数复制模型层

### _prepare_adapter_config (static)
- **方法描述**: 准备适配器配置的静态方法
- **参数**:
  - peft_config: 适配器配置
  - model_config: 模型配置
- **返回值**: 准备好的配置
- **使用案例**: 自动推断目标模块或验证配置完整性

## 模块创建和替换方法

### _create_and_replace
- **方法描述**: 用适配器层创建并替换目标模块的核心方法
- **参数**:
  - lora_config: LoRA 配置
  - adapter_name (`str`): 适配器名称
  - target: 目标模块
  - target_name (`str`): 目标模块名称
  - parent: 父模块
  - current_key: 当前键
  - parameter_name (`Optional[str]`, 默认为 `None`): 参数名称
- **返回值**: `None`
- **使用案例**: LoRA 注入过程中的核心替换逻辑

#### method 解读
该函数是LoRA模块注入的核心逻辑，通过模式匹配确定适配器参数，并根据目标模块类型决定更新或替换策略。

```python
def _create_and_replace(self, lora_config, adapter_name, target, target_name, parent, current_key, *, parameter_name=None):
    # 1. 输入验证
    if current_key is None:
        raise ValueError("Current Key shouldn't be `None`")

    # 2. 模式匹配 - 基于当前键名匹配配置中的rank和alpha模式
    # Regexp matching - Find key which matches current target_name in patterns provided
    r_key = get_pattern_key(lora_config.rank_pattern.keys(), current_key)
    alpha_key = get_pattern_key(lora_config.alpha_pattern.keys(), current_key)
    r = lora_config.rank_pattern.get(r_key, lora_config.r)
    alpha = lora_config.alpha_pattern.get(alpha_key, lora_config.lora_alpha)

    # 3. 构建适配器创建参数
    # 处理不同量化方法和设备配置
    kwargs = {
        "r": r,                                      # LoRA秩
        "lora_alpha": alpha,                           # 缩放因子
        "lora_dropout": lora_config.lora_dropout,        # Dropout率
        "fan_in_fan_out": lora_config.fan_in_fan_out,    # 扇入扇出模式
        "init_lora_weights": lora_config.init_lora_weights, # 权重初始化方法
        "use_rslora": lora_config.use_rslora,           # 使用RSLoRA
        "use_dora": lora_config.use_dora,               # 使用DoRA
        "use_qalora": lora_config.use_qalora,           # 使用QA-LoRA
        "qalora_group_size": lora_config.qalora_group_size, # QA-LoRA分组大小
        "ephemeral_gpu_offload": lora_config.runtime_config.ephemeral_gpu_offload,
        "lora_bias": lora_config.lora_bias,             # LoRA偏置设置
        "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),  # 8bit量化检查
        "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),  # 4bit量化检查
        "parameter_name": parameter_name,
    }

    # 4. 处理torchao合并所需的张量子类
    # for torchao merging, we need the get_apply_tensor_subclass from the quantization config
    try:
        kwargs["get_apply_tensor_subclass"] = operator.attrgetter(
            "hf_quantizer.quantization_config.get_apply_tensor_subclass"
        )(self.model)
    except AttributeError:
        pass  # 如果没有量化配置则忽略

    # 5. 处理不同的量化方法配置
    # 对于GPTQ、AQLM、AWQ等量化方法，获取对应的量化配置
    quant_methods = ["gptq", "aqlm", "awq"]
    for quant_method in quant_methods:
        quantization_config = get_quantization_config(self.model, method=quant_method)
        if quantization_config is not None:
            kwargs[f"{quant_method}_quantization_config"] = quantization_config

    # 6. 检查目标模块类型并决定处理策略
    # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
    from peft.tuners.adalora import AdaLoraLayer

    # 检查是否为需要包装的参数目标
    wrap_target_param = isinstance(target, ParamWrapper) and (adapter_name in target.lora_A)

    if isinstance(target, LoraLayer) and not isinstance(target, AdaLoraLayer) and not wrap_target_param:
        # 6a. 目标已是LoRA层 - 直接更新层参数
        # 当目标模块是已经存在的LoRA层但不是AdaLooraLayer时，直接更新其配置
        target.update_layer(
            adapter_name,
            r,
            lora_alpha=alpha,
            lora_dropout=lora_config.lora_dropout,
            init_lora_weights=lora_config.init_lora_weights,
            use_rslora=lora_config.use_rslora,
            use_dora=lora_config.use_dora,
            lora_bias=lora_config.lora_bias,
        )
    else:
        # 6b. 目标不是LoRA层 - 创建新的LoRA模块
        # 验证参数包装器，避免重复 targeting
        if isinstance(target, ParamWrapper) and (parameter_name == target.parameter_name):
            raise ValueError(
                "Trying to target same nn.Parameter twice, this should not happen. Please open an issue on PEFT repo."
            )

        # 获取模型设备映射用于分布式加载
        device_map = self.model.hf_device_map if hasattr(self.model, "hf_device_map") else None

        # 创建新的LoRA模块
        new_module = self._create_new_module(lora_config, adapter_name, target, device_map=device_map, **kwargs)

        # 如果是额外的适配器，默认不可训练
        if adapter_name not in self.active_adapters:
            new_module.requires_grad_(False)

        # 替换原始模块
        self._replace_module(parent, target_name, new_module, target)
```

### _replace_module
- **方法描述**: 替换模块的辅助方法
- **参数**:
  - parent: 父模块
  - child_name (`str`): 子模块名称
  - new_module: 新模块
  - child: 原子模块
- **返回值**: `None`
- **使用案例**: 在适配器注入过程中安全地替换模块

#### method 解读
该函数负责安全地替换模块，并处理设备分配和元设备管理。

```python
def _replace_module(self, parent, child_name, new_module, child):
    # 1. 基本模块替换
    # 使用setattr将父模块中的子模块替换为新模块
    setattr(parent, child_name, new_module)
    # It's not necessary to set requires_grad here, as that is handled by _mark_only_adapters_as_trainable

    # 2. 处理嵌套的LoRA层
    # child layer wraps the original module, unpack it
    if hasattr(child, "base_layer"):
        child = child.base_layer

    # 3. 设备分配管理
    # 设置元设备用于设备检查
    meta = torch.device("meta")

    # dispatch to correct device - 将新模块正确分配到设备
    for name, module in new_module.named_modules():
        # 只处理适配器相关模块
        if (self.prefix in name) or ("ranknum" in name):
            weight = None

            # 处理不同量化方法的权重获取
            if hasattr(child, "qweight"):
                weight = child.qweight                      # GPTQ/AWQ量化权重
            elif hasattr(child, "W_q"):
                weight = child.W_q                          # BNB量化权重
            elif hasattr(child, "weight"):
                weight = child.weight                        # 标准权重
            elif getattr(child, "in_proj_weight", None) is not None:
                weight = child.in_proj_weight                 # 多头注意力投影权重
            else:
                weight = next(child.parameters())             # 第一个参数作为权重

            # 设备分配：如果模块参数不在元设备上，则分配到权重所在设备
            if not any(p.device == meta for p in module.parameters()):
                module.to(weight.device)
```

### _create_new_module(static)
- **方法描述**: 根据目标类型创建新模块的静态方法
- **参数**:
  - lora_config: LoRA 配置
  - adapter_name (`str`): 适配器名称
  - target: 目标模块
  - **kwargs: 额外参数
- **返回值**: 新创建的模块
- **使用案例**: 为不同类型的层创建相应的 LoRA 适配器

#### method 解读
该函数使用调度器模式根据目标模块类型创建适当的LoRA适配器模块。

```python
@staticmethod
def _create_new_module(lora_config, adapter_name, target, **kwargs):
    # 1. 初始化调度器列表
    # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer
    # 调度器顺序很重要，第一个匹配的总是被使用，因此默认层应该最后检查
    dispatchers = []

    # 2. 处理自定义模块
    # Experimental custom LoRA module support
    if lora_config._custom_modules:
        def dynamic_dispatch_func(target, adapter_name, lora_config, **kwargs):
            new_module = None

            # 获取目标模块的基础层
            if isinstance(target, BaseTunerLayer):
                target_base_layer = target.get_base_layer()
            else:
                target_base_layer = target

            # 遍历自定义模块映射
            for key, custom_cls in lora_config._custom_modules.items():
                if isinstance(target_base_layer, key):
                    new_module = custom_cls(target, adapter_name, **kwargs)
                    break

            return new_module

        dispatchers.append(dynamic_dispatch_func)

    # 3. 处理BitsAndBytes量化模块
    # avoid eager bnb import - 避免急切导入
    if is_bnb_available():
        from .bnb import dispatch_bnb_8bit
        dispatchers.append(dispatch_bnb_8bit)

    if is_bnb_4bit_available():
        from .bnb import dispatch_bnb_4bit
        dispatchers.append(dispatch_bnb_4bit)

    # 4. 添加其他量化方法的调度器
    # 按优先级顺序添加各种量化方法的调度器
    dispatchers.extend([
        dispatch_eetq,      # EETQ量化
        dispatch_aqlm,     # AQLM量化
        dispatch_awq,       # AWQ量化
        dispatch_gptq,      # GPTQ量化
        dispatch_hqq,       # HQQ量化
        dispatch_inc,       # INC量化
        dispatch_torchao,   # TorchAO量化
        dispatch_megatron,  # Megatron LM
        dispatch_default,    # 默认处理（必须最后）
    ])

    # 5. 依次尝试调度器
    new_module = None
    for dispatcher in dispatchers:
        new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
        if new_module is not None:  # first match wins
            break

    # 6. 错误处理
    if new_module is None:
        # no module could be matched
        raise ValueError(
            f"Target module {target} is not supported. Currently, only the following modules are supported: "
            "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv1d`, `torch.nn.Conv2d`, `torch.nn.Conv3d`, "
            "`transformers.pytorch_utils.Conv1D`, `torch.nn.MultiheadAttention.`."
        )

    return new_module
```

## 适配器管理方法

### delete_adapter
- **方法描述**: 删除现有适配器
- **参数**:
  - adapter_name (`str`): 要删除的适配器名称
- **返回值**: `None`
- **使用案例**:
```python
# 删除名为 "adapter1" 的适配器
lora_model.delete_adapter("adapter1")
```

### merge_and_unload
- **方法描述**: 将 LoRA 层合并到基础模型中并卸载适配器，返回独立的模型
- **参数**:
  - progressbar (`bool`, 默认为 `False`): 是否显示进度条
  - safe_merge (`bool`, 默认为 `False`): 是否激活安全合并检查
  - adapter_names (`List[str]`, 可选): 要合并的适配器名称列表，如果为 None 则合并所有活跃适配器
- **返回值**: `torch.nn.Module` - 合并后的模型
- **使用案例**:
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
model = PeftModel.from_pretrained(base_model, peft_model_id)
merged_model = model.merge_and_unload()
```

### unload
- **方法描述**: 移除所有 LoRA 模块而不合并，返回原始基础模型
- **参数**: 无
- **返回值**: `torch.nn.Module` - 原始基础模型
- **使用案例**:
```python
# 获取原始模型，不保留 LoRA 修改
original_model = lora_model.unload()
```

### _unload_and_optionally_merge
- **方法描述**: 卸载并可选择合并模型的内部方法
- **参数**:
  - merge (`bool`, 默认为 `True`): 是否合并
  - progressbar (`bool`, 默认为 `False`): 是否显示进度条
  - safe_merge (`bool`, 默认为 `False`): 是否安全合并
  - adapter_names (`Optional[list[str]]`, 默认为 `None`): 适配器名称列表
- **返回值**: `torch.nn.Module` - 处理后的模型
- **使用案例**: 内部方法，用于 merge_and_unload 和 unload 的实现

### _mark_only_adapters_as_trainable
- **方法描述**: 标记只有适配器参数为可训练，冻结其他所有参数
- **参数**:
  - model (`nn.Module`): 要处理的模型
- **返回值**: `None`
- **使用案例**: 在适配器注入完成后，确保只有 LoRA 参数参与训练

#### method 解读
该函数负责设置模型参数的训练状态，确保只有 LoRA 适配器相关的参数可以训练，而原始预训练模型的参数保持冻结状态。

```python
def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
    # 1. 冻结所有非适配器参数
    # 遍历模型的所有参数，将不属于适配器的参数设置为不可训练
    for n, p in model.named_parameters():
        # 检查参数名称中是否包含适配器前缀
        if self.prefix not in n:
            p.requires_grad = False  # 冻结非适配器参数

    # 2. 处理偏置参数的训练状态
    # 遍历所有活跃的适配器，根据偏置配置设置相应参数的训练状态
    for active_adapter in self.active_adapters:
        # 获取当前适配器的偏置配置
        bias = self.peft_config[active_adapter].bias

        # 如果偏置配置为 "none"，跳过偏置处理
        if bias == "none":
            continue

        # 3. "all" 偏置模式 - 所有偏置参数都可训练
        if bias == "all":
            # 遍历模型的所有参数，将所有包含 "bias" 的参数设置为可训练
            for n, p in model.named_parameters():
                if "bias" in n:
                    p.requires_grad = True

        # 4. "lora_only" 偏置模式 - 只有 LoRA 层的偏置参数可训练
        elif bias == "lora_only":
            # 遍历模型的所有模块，只设置 LoRA 层的偏置参数为可训练
            for m in model.modules():
                # 检查是否为 LoRA 层且具有偏置参数
                if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad = True

        # 5. 不支持的偏置模式 - 抛出错误
        else:
            # 如果偏置配置不是预定义的值，抛出未实现错误
            raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")
```

**参数冻结策略：**
- **前缀匹配**：通过 `self.prefix` 识别适配器参数
- **全局冻结**：默认冻结所有参数，然后选择性解冻

**偏置参数处理：** `none`（无偏置）、`all`（全部偏置）、`lora_only`（仅 LoRA 偏置）

## 加权适配器方法

### _check_add_weighted_adapter
- **方法描述**: 检查 add_weighted_adapter 参数是否有效且与底层模型兼容的辅助方法
- **参数**:
  - adapters (`list[str]`): 适配器名称列表
  - combination_type (`str`): 组合类型
  - svd_rank (`int | None`): SVD 秩
- **返回值**: `tuple[str, int, str]` - (组合类型, 新秩, 密度)
- **使用案例**: 在添加加权适配器前验证参数

### _generalized_task_arithmetic_weighted_adapter
- **方法描述**: 广义任务算术加权适配器的内部实现
- **参数**:
  - combination_type: 组合类型
  - adapters: 适配器列表
  - weights: 权重列表
  - target: 目标模块
  - density: 密度参数
  - majority_sign_method: 多数符号方法
- **返回值**: LoRA 权重增量
- **使用案例**: 实现 linear, ties, dare_linear, dare_ties, magnitude_prune 等组合策略

### _svd_generalized_task_arithmetic_weighted_adapter
- **方法描述**: 基于 SVD 的广义任务算术加权适配器实现
- **参数**:
  - combination_type: 组合类型
  - adapters: 适配器列表
  - weights: 权重列表
  - new_rank (`int`): 新秩
  - target: 目标模块
  - target_lora_A: 目标 LoRA A
  - target_lora_B: 目标 LoRA B
  - density: 密度参数
  - majority_sign_method: 多数符号方法
  - clamp: 限制值
  - full_matrices (`bool`, 默认为 `True`): 是否使用完整矩阵
  - driver: SVD 驱动程序
- **返回值**: 处理后的 LoRA A 和 B 权重
- **使用案例**: 实现 svd, ties_svd, dare_linear_svd 等 SVD 基础的组合策略

## 初始化和转换方法

### subtract_mutated_init
- **方法描述**: 通过比较输出状态字典中的 PiSSA/CorDA/OLoRA 适配器参数与 `adapter_name` 中的初始值，计算 PiSSA/CorDA/OLoRA 的更新，从而将 PiSSA/CorDA/OLoRA 转换为 LoRA
- **参数**:
  - output_state_dict (`dict[str, torch.Tensor]`): 输出状态字典
  - adapter_name (`str`): 适配器名称
  - kwargs: 额外参数
- **返回值**: `dict[str, torch.Tensor]` - 转换后的 LoRA 张量
- **使用案例**: 将高级初始化方法转换为标准 LoRA 格式

## 工具和辅助方法

### __getattr__
- **方法描述**: 获取属性的特殊方法
- **参数**:
  - name (`str`): 属性名称
- **返回值**: 属性值
- **使用案例**: 动态属性访问

### get_peft_config_as_dict
- **方法描述**: 获取 PEFT 配置的字典表示
- **参数**:
  - inference (`bool`, 默认为 `False`): 是否为推理模式
- **返回值**: `dict` - 配置字典
- **使用案例**: 序列化配置或检查当前设置

### _enable_peft_forward_hooks
- **方法描述**: 启用 PEFT 前向钩子
- **参数**:
  - *args: 位置参数
  - **kwargs: 关键字参数
- **返回值**: `None`
- **使用案例**: 在训练过程中启用适配器的特殊处理

### _check_merge_allowed
- **方法描述**: 检查是否允许合并适配器的内部方法
- **参数**: 无
- **返回值**: `None`
- **使用案例**: 在合并操作前验证模型状态和兼容性