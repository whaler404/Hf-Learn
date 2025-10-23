# BaseTunerLayer 概述

`BaseTunerLayer` 是一个调优器层的混入类，为所有调优器提供通用方法和属性。

## 类的描述

`BaseTunerLayer` 是 PEFT 库中所有适配器层的抽象基类，提供了适配器管理的通用接口。它定义了适配器的激活、禁用、合并、删除等核心功能，以及设备管理和数据类型转换等辅助方法。该类作为所有具体适配器实现（如 LoRA 层、AdaLoRA 层等）的基础框架。

## 类属性

- **adapter_layer_names** (`tuple[str, ...]`, 默认值: `()`):
  - 描述：可能包含适配器（可训练）权重的所有层名称的元组。

- **other_param_names** (`tuple[str, ...]`, 默认值: `()`):
  - 描述：可能包含适配器相关参数的其他参数名称的元组。

- **_disable_adapters** (`bool`, 默认值: `False`):
  - 描述：指示是否应禁用所有适配器的内部标志。

- **_active_adapter** (`str | list[str]`, 默认值: `"default"`):
  - 描述：当前活跃的适配器名称，可以是字符串或字符串列表。

- **merged_adapters** (`list[str]`, 默认值: `[]`):
  - 描述：所有已合并适配器的名称列表。

# method 概述

## 核心适配器管理方法

### get_base_layer
- 描述：（递归地）获取基础层。这在调优器层包装另一个调优器层的情况下是必需的。
- 输入参数: 无
- 输出参数: `nn.Module` - 基础层

### enable_adapters
- 描述：切换适配器的启用和禁用状态，负责设置适配器权重的 requires_grad 标志。
- 输入参数:
  - enabled (`bool`): 为 `True` 启用适配器，为 `False` 禁用适配器
- 输出参数: `None`

### set_adapter
- 描述：设置活跃适配器。此外，此函数会将指定适配器设置为可训练（即 requires_grad=True）。
- 输入参数:
  - adapter_names (`str | list[str]`): 要激活的适配器名称
- 输出参数: `None`

### delete_adapter
- 描述：从层中删除适配器。应在所有适配器层上调用此方法，否则将导致状态不一致。如果删除的适配器是活跃适配器，此方法还将设置新的活跃适配器。
- 输入参数:
  - adapter_name (`str`): 要删除的适配器名称
- 输出参数: `None`

## 适配器合并相关方法

### merge
- 描述：合并适配器权重到基础层中（抽象方法，需要在子类中实现）。
- 输入参数:
  - safe_merge (`bool`, 默认值: `False`): 是否进行安全合并
  - adapter_names (`Optional[list[str]]`, 默认值: `None`): 要合并的适配器名称列表
- 输出参数: `None`

### unmerge
- 描述：取消合并适配器权重（抽象方法，需要在子类中实现）。
- 输入参数: 无
- 输出参数: `None`

## 属性访问器

### weight (property)
- 描述：获取基础层的权重。这对于某些 transformers 代码是必需的，例如对于 T5，权重通过 `self.wo.weight` 访问，其中 "wo" 是适配器层。
- 输入参数: 无
- 输出参数: `torch.Tensor` - 基础层权重

### bias (property)
- 描述：获取基础层的偏置。
- 输入参数: 无
- 输出参数: `torch.Tensor` - 基础层偏置

### merged (property)
- 描述：检查是否有适配器已被合并。
- 输入参数: 无
- 输出参数: `bool` - 如果有适配器已合并返回 `True`，否则返回 `False`

### disable_adapters (property)
- 描述：获取适配器禁用状态。使用属性确保 disable_adapters 不被直接设置，而是使用 enable_adapters 方法。
- 输入参数: 无
- 输出参数: `bool` - 适配器禁用状态

### active_adapter (property)
- 描述：获取当前活跃适配器。使用属性确保 active_adapter 不被直接设置，而是使用 set_adapter 方法。
- 输入参数: 无
- 输出参数: `str | list[str]` - 活跃适配器名称

### active_adapters (property)
- 描述：获取当前活跃适配器的列表形式。
- 输入参数: 无
- 输出参数: `list[str]` - 活跃适配器名称列表

## 适配器发现和查询方法

### _get_available_adapters
- 描述：返回在此模块上可以找到的所有适配器名称。
- 输入参数: 无
- 输出参数: `set[str]` - 可用适配器名称集合

### _all_available_adapter_names
- 描述：返回所有可用适配器名称的排序列表。
- 输入参数: 无
- 输出参数: `list[str]` - 所有可用适配器名称的排序列表

## 设备和数据类型管理

### _move_adapter_to_device_of_base_layer
- 描述：将指定名称的适配器移动到基础层的设备上。
- 输入参数:
  - adapter_name (`str`): 适配器名称
  - device (`Optional[torch.device]`, 默认值: `None`): 目标设备，如果为 `None` 则使用基础层的设备
- 输出参数: `None`

### _cast_input_dtype
- 描述：是否转换 forward 方法输入的数据类型。通常启用此功能以使输入数据类型与权重数据类型对齐，但可以通过设置 `layer.cast_input_dtype=False` 来禁用。
- 输入参数:
  - x (`None | torch.Tensor`): 输入张量或 None
  - dtype (`torch.dtype`): 目标数据类型
- 输出参数: `None | torch.Tensor` - 转换后的张量或 None