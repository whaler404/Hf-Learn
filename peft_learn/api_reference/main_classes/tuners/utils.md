# PEFT Tuners Utils 工具函数参考

## onload_layer

### method 概述
一个用于修改包含一个或多个调谐器(tuners)和基础层的模块的工具函数，其中任何模块都可能被卸载到CPU或磁盘。在执行某些操作之前，将模块的子模块移动到执行设备上，操作完成后重新分配基础层状态字典（如果该层被卸载到磁盘），最后将参数重新卸载。

如果模块没有卸载的子模块，此函数不执行任何操作。

**输入参数：**
- `layer` (torch.nn.Module)：包含要合并的调谐器的层

**输出参数：**
- 生成器对象，用于上下文管理

**method example:**
```python
from peft.tuners.tuners_utils import onload_layer

# 使用示例
model = ...  # 包含调谐器的模型
layer = model.get_layer(0)  # 获取特定层

with onload_layer(layer):
    # 在此上下文中，所有卸载的模块都会被加载到执行设备
    output = layer(input_tensor)
    # 退出上下文后，模块会自动卸载回原位置
```

### method 解读
该函数主要用于处理模型层的设备管理和内存优化，特别是对于大型模型中可能被卸载到CPU或磁盘的层。通过上下文管理器确保在执行操作时模块位于正确的设备上，操作完成后恢复到原始状态。

```python
@contextmanager
def onload_layer(layer):
    # 存储需要处理的卸载模块列表
    offloaded_modules = []

    # 遍历所有子模块，处理调谐器模块
    # 跳过根模块("")和基础层("base_layer)，只处理实际的调谐器模块
    for name, module in layer.named_modules():
        if name in ["", "base_layer"]:
            continue

        # 筛选目标类型：具有AlignDevicesHook且处于offload状态的模块
        # 这些是通过accelerate库进行设备管理的模块，包括LoRA、AdaLoRA等调谐器层
        if hasattr(module, "_hf_hook") and isinstance(module._hf_hook, AlignDevicesHook) and module._hf_hook.offload:
            # 将卸载的模块预加载到执行设备
            module._hf_hook.pre_forward(module)
            offloaded_modules.append(module)

    # 检查基础层是否也被卸载
    base_layer_offload = False
    if hasattr(layer, "base_layer") and (
        hasattr(layer.base_layer, "_hf_hook")
        and isinstance(layer.base_layer._hf_hook, AlignDevicesHook)
        and layer.base_layer._hf_hook.offload
    ):
        # 当layer为具有base_layer的调谐器层时（如LoraLayer、AdaLoraLayer等）
        # 处理磁盘卸载的特殊情况
        if torch.device("meta") in layer.base_layer._hf_hook.original_devices.values() and hasattr(
            layer.base_layer._hf_hook.weights_map, "dataset"
        ):
            # 获取磁盘卸载索引，该索引映射模块到safetensors文件
            index = layer.base_layer._hf_hook.weights_map.dataset.index
            module_name = list(dict(layer.base_layer._hf_hook.weights_map.dataset).keys())[0]
            file_name = index[module_name]["safetensors_file"]
            base_name_arr = []

            # 提取基础目录名称，构建合并后的safetensors文件名
            for i in os.path.split(file_name):
                if "--" in i:
                    base_name_arr.append(i)
                    break
                base_name_arr.append(i)
            base_name = os.path.join(*base_name_arr)
            safetensors_filename = base_name + "-merged"

        # 将基础层预加载到执行设备
        layer.base_layer._hf_hook.pre_forward(layer.base_layer)
        base_layer_offload = True

    # 生成器yield，在此期间可以执行需要模块在设备上的操作
    yield

    # 恢复所有卸载模块的状态
    for module in offloaded_modules:
        # 将模块重新卸载到原始设备
        module._hf_hook.post_forward(module, torch.tensor([]))

    if base_layer_offload:
        # 重建权重映射（必须在cpu上以通过memmap将参数发送到磁盘）
        layer.base_layer._hf_hook.weights_map = {
            name: param.to("cpu") for name, param in named_module_tensors(layer.base_layer)
        }

        # 如果原始设备是磁盘，将权重映射卸载到磁盘
        if torch.device("meta") in layer.base_layer._hf_hook.original_devices.values() and hasattr(
            layer.base_layer._hf_hook.weights_map, "dataset"
        ):
            # 用合并后的权重重写目录
            offload_state_dict(safetensors_filename, layer.base_layer._hf_hook.weights_map)

        # 将基础层重新卸载到原始设备
        layer.base_layer._hf_hook.post_forward(layer.base_layer, torch.tensor([]))
```

**关键组件说明：**

1. **`AlignDevicesHook`**: accelerate库中的钩子，用于管理设备分配和卸载
2. **`_hf_hook.offload`**: 标记模块是否处于卸载状态
3. **`torch.device("meta")`**: 表示磁盘卸载的特殊设备标记
4. **`pre_forward()`/`post_forward()`**: 钩子的前向和后向处理方法，用于加载和卸载模块

**支持的模块类型：**
- LoRA层 (`LoraLayer`)
- AdaLoRA层 (`AdaLoraLayer`)
- 其他具有base_layer结构的调谐器层
- 通过accelerate管理的任何可卸载模块

## _find_minimal_target_modules

### method 概述
找到足够将目标模块与其他模块分离的最小目标模块集合。有时候可能会传递一个非常大的 target_modules 列表，这会减慢适配器的加载速度（例如从 diffusers 加载时）。可以将这个列表从数百个项目压缩为少量后缀，这些后缀足以将目标模块与其他模块区分开来。

**输入参数：**
- `target_modules` (list[str] | set[str])：目标模块的列表
- `other_module_names` (list[str] | set[str])：其他模块名称的列表。它们不能与目标模块重叠

**输出参数：**
- `set[str]`：足够将目标模块与其他模块分离的最小目标模块集合

**method example:**
```python
from peft.tuners.tuners_utils import _find_minimal_target_modules

# 示例1：基础用法
target_modules = [f"model.decoder.layers.{i}.self_attn.q_proj" for i in range(100)]
target_modules += [f"model.decoder.layers.{i}.self_attn.v_proj" for i in range(100)]
other_module_names = [f"model.encoder.layers.{i}.self_attn.k_proj" for i in range(100)]
result = _find_minimal_target_modules(target_modules, other_module_names)
# 结果: {"q_proj", "v_proj"}

# 示例2：复杂模块路径
target_modules = [
    "transformer.h.0.attn.c_attn",
    "transformer.h.1.attn.c_attn",
    "transformer.h.2.attn.c_proj"
]
other_module_names = [
    "transformer.h.0.mlp.c_fc",
    "transformer.h.1.mlp.c_fc"
]
result = _find_minimal_target_modules(target_modules, other_module_names)
# 可能结果: {"c_attn", "c_proj"}
```

### method 解读
该函数通过算法优化目标模块列表，找到能够区分目标模块和其他模块的最小后缀集合，提高适配器加载效率。

```python
def _find_minimal_target_modules(target_modules, other_module_names):
    # 输入验证：确保target_modules是有效的列表或集合
    if isinstance(target_modules, str) or not target_modules:
        raise ValueError("target_modules should be a list or set of strings.")

    target_modules = set(target_modules)
    if "" in target_modules:
        raise ValueError("target_modules should not contain an empty string.")

    # 处理其他模块名称，确保无重叠
    other_module_names = set(other_module_names)
    if not target_modules.isdisjoint(other_module_names):
        # 如果有重叠，抛出错误提示用户报告GitHub issue
        raise ValueError("target_modules and other_module_names contain common elements...")

    # 假设模块名称各部分由"."分隔
    def generate_suffixes(s):
        # 生成所有可能的后缀：从完整名称到最后一部分
        parts = s.split(".")
        # [".".join(parts[i:]) for i in range(len(parts))][::-1]
        # 例如：["model.decoder.layers.0.self_attn.q_proj"] 生成：
        # ["q_proj", "self_attn.q_proj", "layers.0.self_attn.q_proj", "decoder.layers.0.self_attn.q_proj", "model.decoder.layers.0.self_attn.q_proj"]
        return [".".join(parts[i:]) for i in range(len(parts))][::-1]

    # 为其他模块名称创建反向查找表，快速检查后缀匹配
    # 目的：避免选择与其他模块冲突的后缀
    other_module_suffixes = {suffix for item in other_module_names for suffix in generate_suffixes(item)}

    # 生成所有目标模块的可能后缀映射
    target_modules_suffix_map = {item: generate_suffixes(item) for item in target_modules}

    # 初始化必需的后缀集合
    required_suffixes = set()

    # 排序以确保确定性行为（因为集合没有顺序）
    for item, suffixes in sorted(target_modules_suffix_map.items(), key=lambda tup: tup[1]):
        # 按最短后缀优先的顺序处理目标模块项
        for suffix in suffixes:
            # 如果后缀已存在或匹配其他模块名称，跳过
            if suffix in required_suffixes or suffix in other_module_suffixes:
                continue

            # 检查添加此后缀是否能覆盖该模块
            # 如果当前模块还没有被任何已选后缀覆盖，则添加此后缀
            if not any(item.endswith("." + req_suffix) for req_suffix in required_suffixes):
                required_suffixes.add(suffix)
                break

    # 如果没有找到有效的后缀，返回原始目标模块
    if not required_suffixes:
        return set(target_modules)

    return required_suffixes
```

**算法逻辑说明：**

1. **后缀生成策略**：将模块路径按"."分割，生成从最后一部分到完整路径的所有可能后缀
2. **冲突检测**：生成其他模块的所有后缀，确保选择的后缀不会与其他模块冲突
3. **贪心选择**：优先选择最短的有效后缀，确保覆盖所有目标模块
4. **覆盖检查**：确保每个目标模块至少被一个选择的后缀覆盖

**使用场景：**
- LoRA适配器配置优化
- 大型模型的目标模块列表压缩
- 提高适配器加载和匹配速度

**时间复杂度分析：**
- 后缀生成：O(n*m)，其中n是模块数量，m是平均模块路径长度
- 冲突检测：O(n*m)
- 后缀选择：O(n²*m)，其中n是目标模块数量

## check_target_module_exists

### method 概述
一个辅助方法，用于检查传递的模块键名是否与适配器配置中的任何目标模块匹配。该方法支持多种匹配模式，包括直接匹配、后缀匹配、正则表达式匹配，以及基于层索引的精确匹配。

**输入参数：**
- `config` (LoraConfig | LycorisConfig)：用于匹配目标模块的配置对象
- `key` (str)：要在配置中搜索匹配的模块键名

**输出参数：**
- `bool | re.Match[str] | None`：如果键匹配配置中的任何目标模块，则返回True或匹配对象；如果未找到匹配，则返回False或None

**method example:**
```python
from peft.tuners.tuners_utils import check_target_module_exists
from peft import LoraConfig

# 示例1：基本模块匹配
config = LoraConfig(target_modules=["q_proj", "v_proj"])
result1 = check_target_module_exists(config, "model.layers.0.q_proj")  # True
result2 = check_target_module_exists(config, "model.layers.0.k_proj")  # False

# 示例2：排除模块匹配
config = LoraConfig(
    target_modules=["q_proj", "v_proj"],
    exclude_modules=["output"]  # 排除包含"output"的模块
)
result3 = check_target_module_exists(config, "model.output.q_proj")  # False (被排除)

# 示例3：层索引匹配
config = LoraConfig(
    target_modules=["q_proj"],
    layers_to_transform=[0, 2],  # 只处理第0层和第2层
    layers_pattern=["layers"]
)
result4 = check_target_module_exists(config, "model.layers.0.q_proj")  # True
result5 = check_target_module_exists(config, "model.layers.1.q_proj")  # False
```

### method 解读
该函数是PEFT适配器模块匹配的核心逻辑，通过多层筛选机制确定是否应该对特定模块应用适配器。

```python
def check_target_module_exists(config, key: str) -> bool | re.Match[str] | None:
    # 第一层筛选：排除模块检查
    # 当配置有exclude_modules属性时，优先进行排除判断
    if hasattr(config, "exclude_modules") and config.exclude_modules:
        if isinstance(config.exclude_modules, str):
            # 如果exclude_modules是字符串，使用正则表达式完整匹配
            if re.fullmatch(config.exclude_modules, key):
                return _ExcludedModule()  # 返回排除模块对象
        elif key in config.exclude_modules:
            # 直接字符串匹配
            return _ExcludedModule()
        elif any(key.endswith(f".{exclude_key}") for exclude_key in config.exclude_modules):
            # 后缀匹配：检查模块路径是否以排除键结尾
            return _ExcludedModule()

    # 第二层筛选：modules_to_save检查
    # 适配器不应匹配要保存的模块，避免ModulesToSaveWrapper内部与潜在适配器之间的行为冲突
    modules_to_save = getattr(config, "modules_to_save", None)
    if modules_to_save:
        if any(re.match(rf"(^|.*\.){m}($|\..*)", key) for m in modules_to_save):
            return _ExcludedModule()

    # 第三层筛选：target_modules和target_parameters兼容性检查
    if (config.target_modules is None) and (config.target_parameters is not None):
        # 如果没有指定target_modules但有target_parameters，这是允许的情况
        return False

    # 第四层筛选：目标模块匹配
    if isinstance(config.target_modules, str):
        # 字符串模式：使用外部辅助函数进行正则表达式匹配
        target_module_found = match_target_against_key(config.target_modules, key)
    elif key in config.target_modules:
        # 直接匹配：模块键在target_modules列表中
        target_module_found = True
    else:
        # 后缀匹配：检查模块路径是否以任何目标键结尾
        target_module_found = any(key.endswith(f".{target_key}") for target_key in config.target_modules)

        # 第五层筛选：层索引精确匹配（仅在使用层索引时）
        layer_indexes = getattr(config, "layers_to_transform", None)
        layers_pattern = getattr(config, "layers_pattern", None)

        # 检查是否使用层索引
        is_using_layer_indexes = layer_indexes is not None and (
            len(layer_indexes) != 0 if isinstance(layer_indexes, list) else True
        )

        if is_using_layer_indexes and target_module_found:
            layer_index = None

            # 处理层模式：None或空表示任何层模式都可以
            if layers_pattern is None or len(layers_pattern) == 0:
                # 提取模块路径中的层索引：匹配类似 "model.layers.0.xxx" 中的 "0"
                layer_index = re.match(r".*\.[^.]*\.(\d+)\.", key)
            else:
                # 使用指定的层模式进行匹配
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern
                for pattern in layers_pattern:
                    # 动态构建正则表达式：rf".*\.{pattern}\.(\d+)\."
                    # 例如pattern="layers"匹配 "model.layers.0.xxx"
                    layer_index = re.match(rf".*\.{pattern}\.(\d+)\.", key)
                    if layer_index is not None:
                        break

            if layer_index is None:
                # 如果没有找到层索引，则模块不匹配
                target_module_found = False
            else:
                # 将找到的层索引转换为整数
                layer_index = int(layer_index.group(1))
                if isinstance(layer_indexes, int):
                    # 单个层索引：检查是否相等
                    target_module_found = layer_index == layer_indexes
                else:
                    # 层索引列表：检查是否在列表中
                    target_module_found = layer_index in layer_indexes

    return target_module_found
```

**hasattr检查的目的说明：**

1. **`hasattr(config, "exclude_modules")`**: 筛选具有排除模块配置的适配器配置
   - 目标类型：LoraConfig、LycorisConfig等具有exclude_modules属性的配置
   - 目的：确定是否需要进行模块排除判断

2. **`getattr(config, "modules_to_save", None)`**: 安全获取要保存的模块列表
   - 目标类型：具有modules_to_save属性的配置
   - 目的：避免与ModulesToSaveWrapper产生冲突

3. **`getattr(config, "layers_to_transform", None)`**: 获取层变换配置
   - 目标类型：支持层索引控制的适配器配置
   - 目的：实现基于层索引的精确模块选择

4. **`getattr(config, "layers_pattern", None)`**: 获取层模式配置
   - 目标类型：具有自定义层模式支持的配置
   - 目的：支持灵活的层路径匹配模式

**关键返回值说明：**
- `True`: 模块匹配，应该应用适配器
- `False`: 模块不匹配，不应该应用适配器
- `_ExcludedModule()`: 模块被明确排除，不应用适配器
- `re.Match[str]`: 正则表达式匹配结果（由match_target_against_key返回）

## inspect_matched_modules

### method 概述
一个辅助函数，用于检查PEFT模型和给定适配器的已匹配和未匹配模块集合。该函数遍历模型中的所有模块，并根据适配器配置将它们分为匹配和未匹配两类。

**输入参数：**
- `tuner` (BaseTuner)：PEFT调谐器实例
- `adapter_name` (str)：适配器名称，默认为"default"

**输出参数：**
- `dict`：包含两个键的字典：
  - `"matched"`：匹配模块的键名列表
  - `"unmatched"`：未匹配模块的键名列表

**method example:**
```python
from peft import get_peft_model, LoraConfig
from peft.tuners.tuners_utils import inspect_matched_modules

# 示例：检查LoRA模型的模块匹配情况
base_model = ...  # 基础模型
config = LoraConfig(target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)

# 检查匹配的模块
result = inspect_matched_modules(model, "default")
print(f"匹配的模块数量: {len(result['matched'])}")
print(f"未匹配的模块数量: {len(result['unmatched'])}")
print("匹配的模块示例:", result['matched'][:5])
```

### method 解读
该函数通过遍历模型的所有模块并使用适配器的匹配逻辑来检查每个模块是否应该应用适配器。

```python
def inspect_matched_modules(tuner: BaseTuner, adapter_name: str = "default") -> dict:
    # 获取指定适配器的配置
    config = tuner.peft_config[adapter_name]

    # 获取模型中所有模块的键名列表
    # tuner.model.named_modules() 返回 (模块名, 模块对象) 的生成器
    key_list = [key for key, _ in tuner.model.named_modules()]

    # 初始化结果字典
    module_dict = {"matched": [], "unmatched": []}

    # 遍历每个模块键名，检查是否匹配适配器配置
    for key in key_list:
        # 使用调谐器内部的检查函数判断模块是否匹配
        if tuner._check_target_module_exists(config, key):
            module_dict["matched"].append(key)    # 匹配的模块
        else:
            module_dict["unmatched"].append(key)  # 未匹配的模块

    return module_dict
```

**处理逻辑说明：**
- **模块遍历**：处理模型中的所有模块，包括嵌套模块
- **匹配检查**：使用调谐器内部的`_check_target_module_exists`方法
- **结果分类**：将模块分为匹配和未匹配两类，便于调试和分析

## _maybe_include_all_linear_layers

### method 概述
辅助函数，如果提供了"all-linear"作为目标模块，则将`target_modules`更新为所有线性/Conv1D层。该函数改编自QLoRA仓库，用于快速配置适配器应用于模型中的所有线性层。

**输入参数：**
- `peft_config` (PeftConfig)：PEFT配置对象
- `model` (nn.Module)：PyTorch模型

**输出参数：**
- `PeftConfig`：更新后的配置对象

**method example:**
```python
from peft import LoraConfig
from peft.tuners.tuners_utils import _maybe_include_all_linear_layers
import transformers

# 示例1：使用"all-linear"快捷方式
config = LoraConfig(target_modules="all-linear")  # 使用快捷方式
model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

# 自动将"all-linear"转换为实际的线性层名称
updated_config = _maybe_include_all_linear_layers(config, model)
print(f"找到的线性层: {list(updated_config.target_modules)[:10]}...")

# 示例2：不处理非"all-linear"的配置
config2 = LoraConfig(target_modules=["q_proj", "v_proj"])  # 指定模块
result_config2 = _maybe_include_all_linear_layers(config2, model)
# 配置保持不变
```

### method 解读
该函数通过智能识别模型中的线性层来自动配置适配器的目标模块。

```python
def _maybe_include_all_linear_layers(peft_config: PeftConfig, model: nn.Module) -> PeftConfig:
    # 检查配置是否有target_modules属性
    if not hasattr(peft_config, "target_modules"):
        return peft_config

    # 检查target_modules是否为"all-linear"字符串
    if not (
        isinstance(peft_config.target_modules, str)
        and peft_config.target_modules.lower() == INCLUDE_LINEAR_LAYERS_SHORTHAND
    ):
        return peft_config

    # 定义线性层的类型和名称模式
    linear_classes = (torch.nn.Linear, Conv1D)  # 标准线性层类型
    linear_names = ("Linear",)                  # 线性层类名模式
    linear_module_names = set()

    # 遍历模型中的所有模块，寻找线性层
    for name, module in model.named_modules():
        # 方式1：基于类型匹配 - 直接的线性层类
        if isinstance(module, linear_classes):
            linear_module_names.add(name)
        elif isinstance(module, BaseTunerLayer) and any(n in type(module).__name__ for n in linear_names):
            # 方式2：基于名称匹配 - 适配器层中的线性层
            # 如果模型已经应用了适配器层，那么"线性层"实际上是适配器层
            # 例如：lora.Linear，而不是nn.Linear
            # 依赖类名而非类型检查，因为PEFT方法种类繁多，类型列表会很快过时
            # 按约定，PEFT中的线性层类名类似"Linear", "Linear4bit", "HqqLoraLinear"等
            linear_module_names.add(name)

    # 尽可能移除不应该作为目标的线性层
    module_names_to_exclude = set()

    # 处理预训练模型的特殊层
    if isinstance(model, PreTrainedModel):
        output_emb = model.get_output_embeddings()
        if output_emb is not None:
            # 忽略文本生成模型的最后一个分类头
            last_module_name = [name for name, module in model.named_modules() if module is output_emb][0]
            module_names_to_exclude.add(last_module_name)
        elif peft_config.task_type == TaskType.SEQ_CLS:
            # 忽略分类模型的分类头（issue 2027）
            # 检查常见的分类头名称
            for name in SEQ_CLS_HEAD_NAMES:
                cls_head = getattr(model, name, None)
                if cls_head is not None:
                    last_module_name = [name for name, module in model.named_modules() if module is cls_head][0]
                    module_names_to_exclude.add(last_module_name)
                    break

    # 避免嵌套的LoRA层，即LoRA应用于已存在的lora_A, lora_B等
    for prefix, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            for suffix, child in module.named_modules():
                if suffix:
                    # 排除适配器层的内部模块，避免重复应用适配器
                    module_names_to_exclude.add(f"{prefix}.{suffix}")

    # 从目标模块中移除需要排除的模块
    linear_module_names -= module_names_to_exclude
    peft_config.target_modules = linear_module_names
    return peft_config
```

**hasattr检查的目的说明：**

1. **`hasattr(peft_config, "target_modules")`**: 筛选具有目标模块配置的PEFT配置
   - 目标类型：LoraConfig、AdaLoraConfig等支持target_modules的配置
   - 目的：确定是否需要进行自动线性层发现

2. **`isinstance(model, PreTrainedModel)`**: 筛选Hugging Face预训练模型
   - 目标类型：transformers库中的预训练模型
   - 目的：应用特殊的排除规则（如输出嵌入层、分类头）

3. **`getattr(model, name, None)`**: 安全获取模型的特定组件
   - 目标类型：具有分类头或其他特殊层的模型
   - 目的：检查和排除不应该应用适配器的模块

**智能排除机制：**
- 输出嵌入层排除：避免影响文本生成
- 分类头排除：避免影响序列分类任务
- 嵌套适配器排除：避免重复应用适配器

## check_adapters_to_merge

### method 概述
辅助函数，用于检查哪些适配器应该被合并。只返回那些尚未合并的适配器。如果部分或全部适配器已经合并，会给出警告。

**输入参数：**
- `module` (BaseTunerLayer)：调谐器层实例
- `adapter_names` (Optional[list[str]])：要检查的适配器名称列表，默认为None（使用活动适配器）

**输出参数：**
- `list[str]`：需要合并的适配器名称列表（排除已合并的适配器）

**method example:**
```python
from peft.tuners.tuners_utils import check_adapters_to_merge
import warnings

# 示例：检查适配器合并状态
lora_layer = ...  # LoRA层实例

# 示例1：检查所有活动适配器
adapters_to_merge = check_adapters_to_merge(lora_layer)
print(f"需要合并的适配器: {adapters_to_merge}")

# 示例2：检查指定适配器
adapters_to_merge = check_adapters_to_merge(lora_layer, ["adapter1", "adapter2"])

# 示例3：处理已合并的适配器（会触发警告）
with warnings.catch_warnings(record=True) as w:
    adapters_to_merge = check_adapters_to_merge(lora_layer, ["already_merged_adapter"])
    if w:
        print(f"警告: {w[0].message}")
```

### method 解读
该函数通过检查适配器的合并状态来避免重复合并操作。

```python
def check_adapters_to_merge(module: BaseTunerLayer, adapter_names: Optional[list[str]] = None) -> list[str]:
    # 如果未指定适配器名称，使用模块的活动适配器
    if adapter_names is None:
        adapter_names = module.active_adapters

    # 验证输入类型：必须是字符串列表，不能是单个字符串
    if isinstance(adapter_names, str):
        raise ValueError(f"adapter_names should be a list of strings, got {adapter_names!r}.")

    # 检查模块是否已有合并的适配器
    if module.merged:
        # 获取已合并的适配器集合
        merged_adapters = set(module.merged_adapters)

        # 过滤掉已经合并的适配器，只保留未合并的
        adapter_names = [name for name in adapter_names if name not in merged_adapters]

        # 根据剩余适配器情况发出相应警告
        if adapter_names:
            # 部分适配器已合并，仍有适配器需要合并
            warnings.warn(
                f"Already following adapters were merged {','.join(module.merged_adapters)}. "
                f"You are now additionally merging {','.join(adapter_names)}."
            )
        else:
            # 所有适配器都已合并，无需操作
            warnings.warn("All adapters are already merged, nothing to do.")

    return adapter_names
```

**hasattr/getattr检查的目的说明：**

1. **`module.active_adapters`**: 获取模块的活动适配器列表
   - 目标类型：BaseTunerLayer及其子类（LoraLayer、AdaLoraLayer等）
   - 目的：提供默认的适配器选择，简化调用

2. **`module.merged`**: 检查模块的合并状态
   - 目标类型：支持合并操作的调谐器层
   - 目的：确定是否需要进行合并状态检查

3. **`module.merged_adapters`**: 获取已合并的适配器列表
   - 目标类型：具有合并状态跟踪的调谐器层
   - 目的：避免重复合并已合并的适配器

**状态管理逻辑：**
- **未合并状态**：直接返回所有指定适配器
- **部分合并状态**：返回未合并的适配器，发出部分警告
- **全部合并状态**：返回空列表，发出全部合并警告

## clone_module

### method 概述
克隆PyTorch模型中的模块。可以选择在原始模块和克隆模块之间共享所有参数。简化在操作模型架构时重用模块的过程。

**输入参数：**
- `module` (nn.Module)：要克隆的PyTorch模块
- `share_weights` (bool)：是否在原始模块和克隆模块之间共享参数，默认为False

**输出参数：**
- `nn.Module`：克隆后的模块实例

**method example:**
```python
from peft.tuners.tuners_utils import clone_module
import torch.nn as nn

# 示例1：简单克隆（不共享权重）
original_layer = nn.Linear(768, 768)
cloned_layer = clone_module(original_layer)

# 修改原始层不影响克隆层
original_layer.weight.data.fill_(1.0)
print(f"原始层权重均值: {original_layer.weight.mean().item()}")
print(f"克隆层权重均值: {cloned_layer.weight.mean().item()}")

# 示例2：克隆并共享权重
shared_layer = clone_module(original_layer, share_weights=True)

# 修改原始层会影响共享权重的克隆层
original_layer.weight.data.fill_(2.0)
print(f"共享克隆层权重均值: {shared_layer.weight.mean().item()}")

# 示例3：复杂模块克隆
complex_module = nn.Sequential(
    nn.Linear(768, 512),
    nn.ReLU(),
    nn.Linear(512, 256)
)
cloned_complex = clone_module(complex_module, share_weights=True)
```

### method 解读
该函数通过深度复制创建模块的独立副本，并可选择性地建立参数共享机制。

```python
def clone_module(module: nn.Module, share_weights=False):
    # 首先进行模块的深度复制，创建完全独立的副本
    clone = copy.deepcopy(module)

    # 定义内部函数用于在源模块和目标模块之间共享权重
    def _share_weights(src: nn.Module, dst: nn.Module):
        # 遍历源模块的直接参数（不递归）
        for name, param in src.named_parameters(recurse=False):
            # 在目标模块中注册相同的参数对象，实现权重共享
            dst.register_parameter(name, param)

    # 如果启用权重共享
    if share_weights:
        # 遍历模块及其所有子模块
        for name, submodule in module.named_modules():
            # 获取克隆模块中对应的子模块
            cloned_submodule = clone.get_submodule(name)
            # 在源子模块和克隆子模块之间共享权重
            _share_weights(submodule, cloned_submodule)

    return clone
```

**hasattr/getattr检查的目的说明：**

1. **`module.named_parameters(recurse=False)`**: 获取模块的直接参数
   - 目标类型：任何nn.Module或其子类
   - 目的：只处理模块的直接参数，避免递归处理子模块参数

2. **`module.named_modules()`**: 获取模块及其所有子模块
   - 目标类型：任何nn.Module或其子类
   - 目的：遍历完整的模块层次结构，确保所有层级的参数都能正确共享

3. **`clone.get_submodule(name)`**: 获取克隆模块中的特定子模块
   - 目标类型：深度复制后的模块实例
   - 目的：定位克隆模块中与原始模块对应的子模块，建立参数映射关系

**权重共享机制：**
- **独立克隆**：`share_weights=False`，参数完全独立，修改互不影响
- **共享克隆**：`share_weights=True`，参数对象共享，修改一方会影响另一方
- **递归处理**：处理嵌套的模块层次结构，确保所有子模块都正确处理

**技术特点：**
- 使用`copy.deepcopy`确保模块结构的完全复制
- 通过`register_parameter`建立参数引用，而非值复制
- 递归处理模块层次，支持复杂的神经网络架构

## replicate_layers

### method 概述
在transformer模型中复制层并实现权重共享。该函数在`model[(.model)*].layers`路径查找模块列表属性，并根据层映射复制模块列表中的层。例如，映射`[[0, 4], [2, 5]]`将取层集合`[0, 1, 2, 3, 4]`并将其替换为包含`[0, 1, 2, 3, 2, 3, 4]`的模块列表。

**输入参数：**
- `model` (nn.Module)：要复制层的transformer模型
- `layer_map` (list[tuple[int, int]])：层映射列表，每个元组指定要复制的层的起始和结束索引

**输出参数：**
- `None`：直接修改输入模型，无返回值

**method example:**
```python
from peft.tuners.tuners_utils import replicate_layers
import transformers

# 示例1：基本层复制 - 重复中间层
model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
# 原始模型有12层，我们想创建一个更大的模型
layer_map = [
    (0, 6),   # 复制前6层 [0, 1, 2, 3, 4, 5]
    (2, 6),   # 重复复制第2-5层 [2, 3, 4, 5]
    (6, 12)   # 添加后6层 [6, 7, 8, 9, 10, 11]
]
# 最终模型将有：6 + 4 + 6 = 16层
replicate_layers(model, layer_map)

# 示例2：简单重复模式
model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
layer_map = [
    (0, 4),   # [0, 1, 2, 3]
    (2, 4),   # [2, 3] 重复
    (4, 12)   # [4, 5, 6, 7, 8, 9, 10, 11]
]
# 最终层序列：[0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# 示例3：完整架构扩展
model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
layer_map = [
    (0, 12),     # 复制所有原始层
    (6, 12)      # 重复复制后半部分层
]
replicate_layers(model, layer_map)
```

### method 解读
该函数通过识别不同transformer架构的层存储模式，实现跨架构的层复制功能。

```python
def replicate_layers(model: nn.Module, layer_map: list[tuple[int, int]]):
    # 第一步：处理嵌套模型结构
    # 某些模型将主模型包装在model属性中，需要递归查找核心模型
    while hasattr(model, "model"):
        model = model.model

    # 特殊处理：Bert模型变体将主模型嵌套在bert属性下
    if hasattr(model, "bert"):
        model = model.bert

    # 第二步：识别模型架构类型并定位层
    model_type = None
    layers: nn.ModuleList = None

    if hasattr(model, "layers"):
        # Llama架构：model.layers
        model_type = "llama"
        layers = model.layers
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        # Bert架构：model.encoder.layer
        model_type = "bert"
        layers = model.encoder.layer
    elif hasattr(model, "h"):
        # Falcon架构：model.h
        model_type = "falcon"
        layers = model.h

    # 验证是否成功找到支持的层结构
    if not model_type or not isinstance(layers, nn.ModuleList):
        raise ValueError(
            "Could not locate the layers attribute in the model. "
            "Expected Llama, Bert or Falcon compatible architectures."
        )

    # 第三步：根据映射创建新的层列表
    new_layers = []
    for start, end in layer_map:
        # 遍历每个映射区间的层索引
        for i in range(start, end):
            current_idx = len(new_layers)
            # 克隆层并启用权重共享，减少内存使用
            new_layers.append(clone_module(layers[i], share_weights=True))

            # 修复Hugging Face transformers中引入的layer_idx问题
            # 这是解决HF transformers中layer_idx问题的临时方案
            for submodule in new_layers[-1].modules():
                if hasattr(submodule, "layer_idx"):
                    # 更新子模块的层索引以匹配新位置，保持兼容性
                    submodule.layer_idx = current_idx

    # 第四步：根据架构类型替换原始层列表
    layers = nn.ModuleList(new_layers)
    if model_type == "llama":
        model.layers = layers
    elif model_type == "bert":
        model.encoder.layer = layers
    elif model_type == "falcon":
        model.h = layers
    else:
        raise ValueError("Unexpected model type, need to handle post-processing of layers.")

    # 第五步：更新模型配置中的层数
    # 这对Llama、Bert、Falcon等架构都是通用的，保持配置与实际结构同步
    if hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = len(new_layers)
```

**关键特性说明：**

**架构兼容性**：
- **Llama架构**: `model.layers` - 直接层存储
- **Bert架构**: `model.encoder.layer` - encoder嵌套结构，支持bert属性嵌套
- **Falcon架构**: `model.h` - 简化的单字母属性名

**权重共享机制**：
- 通过`clone_module(layers[i], share_weights=True)`实现参数共享
- 复制的层与原始层共享权重，显著减少内存占用
- 支持复杂的层复制模式和重复结构

**层索引管理**：
- 自动更新复制层的`layer_idx`属性
- 确保与Hugging Face transformers的兼容性
- 维护模型内部层索引的一致性

**配置同步**：
- 自动更新`model.config.num_hidden_layers`
- 保持模型配置与实际结构的一致性
- 支持后续的模型保存和加载操作