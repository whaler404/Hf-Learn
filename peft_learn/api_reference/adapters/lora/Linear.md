# Linear

Linear 类是在密集层中实现 LoRA（Low-Rank Adaptation）的核心类，继承自 nn.Module 和 LoraLayer。它为标准 PyTorch 线性层提供 LoRA 适配功能，支持多种 LoRA 变体和初始化方法。

## 核心方法

核心方法：
- [Linear.\_\_init\_\_](#__init__)
    - [LoraLayer.update_layer](LoraLayer.md#update_layer)
- [Linear.forward](#forward)

## 类描述

Linear 类将 LoRA 适配器应用到标准的 `torch.nn.Linear` 层，通过添加可训练的低秩矩阵来高效地微调模型，大幅减少可训练参数数量。

## 参数

- **base_layer** (nn.Module): 要适配的基础线性层，必须是 `torch.nn.Linear` 实例
- **adapter_name** (str): 适配器的名称，用于标识和管理多个适配器
- **r** (int): LoRA 矩阵的秩，默认为 0（表示使用配置中的秩）
- **lora_alpha** (int): LoRA 缩放因子，默认为 1，用于缩放 LoRA 权重
- **lora_dropout** (float): LoRA Dropout 概率，默认为 0.0，用于正则化
- **fan_in_fan_out** (bool): 是否启用扇入扇出模式，默认为 False。当为 True 时，层权重形状存储为 (fan_in, fan_out) 格式
- **is_target_conv_1d_layer** (bool): 标识目标是否为 Conv1D 层，默认为 False
- **init_lora_weights** (Union[bool, str]): LoRA 权重初始化方法，默认为 True。可以是布尔值或字符串（如 "gaussian"、"true" 等）
- **use_rslora** (bool): 是否使用 RS-LoRA（Weighted LoRA）缩放方法，默认为 False
- **use_dora** (bool): 是否使用 DoRA（Weight-Decomposed Low-Rank Adaptation）变体，默认为 False
- **lora_bias** (bool): 是否为 LoRA 适配器添加偏置项，默认为 False
- **kwargs**: 其他关键字参数，用于传递给基类或自定义配置

## 返回值

无返回值，直接修改传入的层或创建新的适配器层。

## 使用案例

```python
import torch.nn as nn
from peft.tuners.lora.layer import Linear

# 基本用法
base_layer = nn.Linear(768, 768)
lora_layer = Linear(
    base_layer=base_layer,
    adapter_name="default",
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    init_lora_weights="gaussian"
)

# 批量数据使用
input_tensor = torch.randn(2, 10, 768)
output = lora_layer(input_tensor)

# 合并适配器
lora_layer.merge(safe_merge=True)

# 取消合并
lora_layer.unmerge()

# 多适配器支持
lora_layer.update_layer(
    adapter_name="task_adapter",
    r=8,
    lora_alpha=16,
    use_rslora=True
)
```

# method

## 初始化和配置方法

### `__init__`
- **方法描述**: 初始化 Linear 实例，设置 LoRA 适配器参数并配置层的属性
- **参数**:
  - `base_layer` (nn.Module): 基础线性层
  - `adapter_name` (str): 适配器名称
  - `r` (int): LoRA 秩，默认为 0
  - `lora_alpha` (int): LoRA alpha，默认为 1
  - `lora_dropout` (float): Dropout 概率，默认为 0.0
  - `fan_in_fan_out` (bool): 扇入扇出模式，默认为 False
  - `is_target_conv_1d_layer` (bool): Conv1D 层标识，默认为 False
  - `init_lora_weights` (Union[bool, str]): 权重初始化方法，默认为 True
  - `use_rslora` (bool): 使用 RS-LoRA，默认为 False
  - `use_dora` (bool): 使用 DoRA，默认为 False
  - `lora_bias` (bool): LoRA 偏置，默认为 False
  - `kwargs`: 其他参数
- **返回值**: None
- **使用案例**:
```python
# 创建带 DoRA 的 Linear LoRA 层
base_layer = nn.Linear(768, 768)
lora_layer = Linear(
    base_layer=base_layer,
    adapter_name="dora_adapter",
    r=16,
    lora_alpha=32,
    use_dora=True,  # 启用 DoRA 变体
    fan_in_fan_out=True,  # 使用扇入扇出模式
    init_lora_weights="true"
)
```

#### method 解读
该函数是 Linear 类的初始化方法，负责创建专门针对线性层的 LoRA 适配器。

```python
def __init__(self, base_layer, adapter_name, r=0, lora_alpha=1, lora_dropout=0.0, fan_in_fan_out=False, is_target_conv_1d_layer=False, init_lora_weights=True, use_rslora=False, use_dora=False, lora_bias=False, **kwargs) -> None:
    # 1. 调用父类初始化
    # 调用 LoraLayer 基类的初始化方法，处理基础层识别和属性设置
    super().__init__()
    LoraLayer.__init__(self, base_layer, **kwargs)

    # 2. 存储线性层特有的配置
    # 保存扇入扇出模式标识
    self.fan_in_fan_out = fan_in_fan_out

    # 3. 设置当前活动适配器
    self._active_adapter = adapter_name

    # 4. 调用通用的 LoRA 层更新方法
    # 这将创建所有的 LoRA 权重矩阵和参数配置
    self.update_layer(
        adapter_name,           # 适配器名称
        r,                   # LoRA 秩
        lora_alpha=lora_alpha, # LoRA alpha
        lora_dropout=lora_dropout, # Dropout 概率
        init_lora_weights=init_lora_weights,  # 权重初始化方法
        use_rslora=use_rslora,     # RS-LoRA 配置
        use_dora=use_dora,           # DoRA 配置
        lora_bias=lora_bias,          # LoRA 偏置配置
    )

    # 5. 存储层类型标识
    # 标识目标层是否为 Conv1D 类型
    self.is_target_conv_1d_layer = is_target_conv_1d_layer
```

### `resolve_lora_variant`
- **方法描述**: 根据配置的 LoRA 变体参数返回对应的变体实现
- **参数**:
  - `use_dora` (bool): 是否使用 DoRA 变体
  - `kwargs`: 其他关键字参数
- **返回值**: `Optional[LoraVariant]` - 返回 DoRA 变体对象或 None
- **使用案例**:
```python
# 检查 DoRA 支持
variant = lora_layer.resolve_lora_variant(use_dora=True)
if variant is not None:
    print(f"DoRA variant: {type(variant).__name__}")
```

## 合并和权重管理方法

### `merge`
- **方法描述**: 将活动适配器的权重合并到基础层权重中，可选择进行安全合并检查
- **参数**:
  - `safe_merge` (bool): 是否启用安全合并，默认为 False。安全合并会检查 NaN 值
  - `adapter_names` (Optional[list[str]]): 要合并的适配器名称列表，默认为 None（合并所有活动适配器）
- **返回值**: None
- **使用案例**:
```python
# 安全合并所有适配器
lora_layer.merge(safe_merge=True)

# 合并特定适配器
lora_layer.merge(adapter_names=["adapter1", "adapter2"])

# 不安全合并（更快）
lora_layer.merge(safe_merge=False)
```

### `unmerge`
- **方法描述**: 从基础层权重中取消合并所有已合并的适配器，恢复原始权重
- **参数**: 无
- **返回值**: None
- **使用案例**:
```python
# 取消合并以继续训练
lora_layer.unmerge()

# 多次调用是安全的
lora_layer.unmerge()  # 会提示已经取消合并
```

### `get_delta_weight`
- **方法描述**: 计算指定适配器的 delta 权重（相对于原始权重的增量）
- **参数**:
  - `adapter` (str): 适配器名称
- **返回值**: `torch.Tensor` - 计算得到的 delta 权重张量
- **使用案例**:
```python
# 获取适配器的权重增量
delta = lora_layer.get_delta_weight("default")
print(f"Delta weight shape: {delta.shape}")

# 用于权重分析和调试
weight_magnitude = delta.norm()
print(f"Delta magnitude: {weight_magnitude.item()}")
```

## 前向传播方法

### `forward`
- **方法描述**: 执行前向传播，根据适配器状态选择不同的计算路径
- **参数**:
  - `x` (torch.Tensor): 输入张量
  - `*args`: 位置参数
  - `adapter_names` (Optional[list[str]]): 指定要使用的适配器名称
  - `**kwargs`: 关键字参数
- **返回值**: `torch.Tensor` - 计算结果
- **使用案例**:
```python
# 基本前向传播
input_tensor = torch.randn(2, 10, 768)
output = lora_layer(input_tensor)

# 使用特定适配器
output = lora_layer(input_tensor, adapter_names=["task_adapter"])

# 混合批次前向传播
# 在同一批次中使用不同的适配器
output = lora_layer(input_tensor, adapter_names=["adapter1", "adapter2"])
```

#### method 解读
该函数是 Linear 类的核心前向传播方法，根据适配器状态执行不同的计算路径。

```python
def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    # 1. 参数检查和前向传播准备
    # 检查前向传播参数是否与模型配置和状态兼容
    self._check_forward_args(x, *args, **kwargs)

    # 获取适配器配置
    adapter_names = kwargs.pop("adapter_names", None)

    # 2. 根据适配器状态选择计算路径
    if self.disable_adapters:
        # 适配器禁用状态
        if self.merged:
            # 如果已合并，先取消合并再使用原始层
            self.unmerge()
        # 直接使用基础层进行前向传播
        result = self.base_layer(x, *args, **kwargs)

    elif adapter_names is not None:
        # 指定了特定适配器名称 - 混合批次前向传播
        result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)

    elif self.merged:
        # 适配器已合并状态 - 只使用基础层
        result = self.base_layer(x, *args, **kwargs)

    else:
        # 默认情况 - 使用活动适配器进行前向传播
        result = self.base_layer(x, *args, **kwargs)

        # 3. LoRA 计算处理
        # 保存原始输出数据类型用于后续恢复
        torch_result_dtype = result.dtype

        # 获取所有活动适配器的键名
        lora_A_keys = self.lora_A.keys()

        # 4. 遍历所有活动适配器并应用 LoRA 计算
        for active_adapter in self.active_adapters:
            # 检查适配器是否存在于 LoRA A 权重字典中
            if active_adapter not in lora_A_keys:
                continue  # 跳过不存在的适配器

            # 获取适配器对应的 LoRA 组件
            lora_A = self.lora_A[active_adapter]        # LoRA A 矩阵
            lora_B = self.lora_B[active_adapter]        # LoRA B 矩阵
            dropout = self.lora_dropout[active_adapter]    # Dropout 层
            scaling = self.scaling[active_adapter]         # 缩放因子

            # 5. 输入类型转换和 LoRA 计算
            # 将输入转换为 LoRA 权重的数据类型
            x = self._cast_input_dtype(x, lora_A.weight.dtype)

            # 检查是否使用 DoRA 变体
            if active_adapter not in self.lora_variant:  # 标准 LoRA 计算路径
                # 标准 LoRA 公式: y = x W^T + x A B^T
                result = result + lora_B(lora_A(dropout(x))) * scaling
            else:
                # DoRA 变体的前向传播
                # DoRA 使用权重分解和幅度向量的特殊计算
                result = self.lora_variant[active_adapter].forward(
                    self,                    # LoraLayer 实例
                    active_adapter=active_adapter,  # 当前适配器名称
                    x=x,                      # 输入张量
                    result=result,              # 基础计算结果
                )

        # 6. 恢复输出数据类型
        result = result.to(torch_result_dtype)

    # 7. 返回计算结果
    return result
```
