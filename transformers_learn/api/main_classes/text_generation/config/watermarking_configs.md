# 水印配置类

GenerationConfig 支持两种水印配置：基础水印配置和 SynthID 文本水印配置。

## BaseWatermarkingConfig

所有水印配置的基类，提供了通用的序列化和配置管理功能。

### 方法

#### `from_dict(config_dict, **kwargs)`
从字典构造水印配置实例。

**参数**:
- `config_dict` (`dict[str, Any]`): 包含配置参数的字典
- `**kwargs`: 用于覆盖字典值的额外关键字参数

**返回**: `BaseWatermarkingConfig` 实例

#### `to_json_file(json_file_path)`
将配置保存到 JSON 文件。

**参数**:
- `json_file_path` (`Union[str, os.PathLike]`): 保存配置的 JSON 文件路径

#### `to_dict()`
将实例序列化为 Python 字典。

**返回**: `dict[str, Any]` - 包含所有配置属性的字典

#### `to_json_string()`
将实例序列化为 JSON 格式字符串。

**返回**: `str` - JSON 格式的配置字符串

#### `update(**kwargs)`
用新值更新配置属性。

**参数**:
- `**kwargs`: 表示配置属性及其新值的关键字参数

## WatermarkingConfig

基础水印配置类，用于在生成的文本中添加水印。

### 参数

#### `greenlist_ratio`
- **类型**: `float`
- **默认值**: `0.25`
- **含义**: "绿色" token 与词汇表大小的比例
- **取值范围**: 0.0-1.0

#### `bias`
- **类型**: `float`
- **默认值**: `2.0`
- **含义**: 添加到选定"绿色"token logits 的偏置值

#### `hashing_key`
- **类型**: `int`
- **默认值**: `15485863` (第百万个素数)
- **含义**: 水印使用的哈希密钥

#### `seeding_scheme`
- **类型**: `str`
- **默认值**: `"lefthash"`
- **可选值**:
  - `"lefthash"` (默认): "绿色" token 选择依赖于前一个 token（论文中的算法 2）
  - `"selfhash"`: "绿色" token 选择依赖于当前 token 本身（论文中的算法 3）

#### `context_width`
- **类型**: `int`
- **默认值**: `1`
- **含义**: 用于种子生成的前 token 上下文长度
- **说明**: 更高的上下文长度使水印更加鲁棒

### 使用示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, WatermarkingConfig

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# 水印配置
watermarking_config = WatermarkingConfig(
    greenlist_ratio=0.25,
    bias=2.0,
    hashing_key=15485863,
    seeding_scheme="lefthash",
    context_width=1
)

# 带水印的生成
inputs = tokenizer("The future of AI is", return_tensors="pt")
outputs = model.generate(
    **inputs,
    watermarking_config=watermarking_config,
    do_sample=True,
    max_new_tokens=50
)
```

## SynthIDTextWatermarkingConfig

Google 的 SynthID 文本水印配置类，提供更高级的水印技术。

### 参数

#### `ngram_len`
- **类型**: `int`
- **默认值**: (必需)
- **含义**: N-gram 长度

#### `keys`
- **类型**: `list[int]`
- **默认值**: (必需)
- **含义**: 水印密钥序列，每个深度一个
- **示例**: `[654, 400, 836, 123, 340, 443, 597, 160, 57]`

#### `context_history_size`
- **类型**: `int`
- **默认值**: `1024`
- **含义**: 跟踪已见上下文的张量大小

#### `sampling_table_seed`
- **类型**: `int`
- **默认值**: `0`
- **含义**: 生成采样表的随机种子

#### `sampling_table_size`
- **类型**: `int`
- **默认值**: `65536` (2^16)
- **含义**: 采样表大小
- **限制**: 必须 < 2^24

#### `skip_first_ngram_calls`
- **类型**: `bool`
- **默认值**: `False`
- **含义**: 是否跳过前 n-gram 调用

#### `debug_mode`
- **类型**: `bool`
- **默认值**: `False`
- **含义**: 调试模式
- **说明**: logits 在应用水印修改前被修改为均匀分布。用于测试实现。

### 使用示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, SynthIDTextWatermarkingConfig

tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b', padding_side="left")
model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b')

# SynthID 文本水印配置
watermarking_config = SynthIDTextWatermarkingConfig(
    keys=[654, 400, 836, 123, 340, 443, 597, 160, 57],
    ngram_len=5,
    context_history_size=1024,
    sampling_table_seed=0,
    sampling_table_size=65536,
    skip_first_ngram_calls=False,
    debug_mode=False,
)

# 带水印的生成
tokenized_prompts = tokenizer(["Once upon a time, "], return_tensors="pt", padding=True)
output_sequences = model.generate(
    **tokenized_prompts,
    watermarking_config=watermarking_config,
    do_sample=True,
    max_new_tokens=10
)
watermarked_text = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
```

## 水印验证

### 检测水印

虽然 GenerationConfig 主要用于生成带水印的文本，但检测水印需要单独的处理：

```python
# 伪代码 - 实际检测需要专门的检测器
def detect_watermark(text, watermarking_config):
    # 实现水印检测逻辑
    # 返回水印存在的概率和置信度
    pass
```

### 水印属性

好的水印应该具备：
1. **不可感知性** - 不影响文本质量
2. **鲁棒性** - 能抵抗轻微修改
3. **唯一性** - 能唯一标识生成者
4. **高效性** - 计算开销小

## 配置比较

| 特性 | WatermarkingConfig | SynthIDTextWatermarkingConfig |
|------|-------------------|------------------------------|
| **复杂度** | 简单 | 高级 |
| **鲁棒性** | 中等 | 高 |
| **性能开销** | 低 | 中等 |
| **配置难度** | 简单 | 复杂 |
| **检测难度** | 中等 | 高 |
| **适用场景** | 一般用途 | 高安全性需求 |

## 注意事项

1. **性能影响**: 水印会增加一些计算开销
2. **文本质量**: 合理的配置应该不明显影响生成质量
3. **兼容性**: 并非所有模型都支持所有水印类型
4. **检测**: 水印检测需要相应的检测工具
5. **密钥管理**: 妥善保管水印密钥以确保安全性

## 更多资源

- [Watermarking 论文](https://huggingface.co/papers/2306.04634)
- [SynthID 论文](https://www.nature.com/articles/s41586-024-08025-4)
- [Hugging Face 水印文档](https://huggingface.co/docs/transformers/main_classes/text_generation#watermarking)