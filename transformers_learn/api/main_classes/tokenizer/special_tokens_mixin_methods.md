# SpecialTokensMixin 方法文档

本文档整理了 `SpecialTokensMixin` 类提供的所有方法，按功能分类说明。`SpecialTokensMixin` 是一个被 `PreTrainedTokenizer` 和 `PreTrainedTokenizerFast` 继承的混入类，用于处理与特殊令牌相关的特定行为。

## 目录

- [初始化和配置方法](#初始化和配置方法)
- [特殊令牌管理方法](#特殊令牌管理方法)
- [令牌添加方法](#令牌添加方法)
- [属性访问和设置方法](#属性访问和设置方法)
- [查询和获取方法](#查询和获取方法)
- [内部和扩展方法](#内部和扩展方法)

---

## 类概述

`SpecialTokensMixin` 类为分词器提供了特殊令牌处理的统一接口。它允许以模型无关的方式直接访问这些特殊令牌，并支持设置和更新特殊令牌。

### 支持的特殊令牌类型

- `bos_token`: 句子开始令牌
- `eos_token`: 句子结束令牌
- `unk_token`: 未知词汇令牌
- `sep_token`: 分隔令牌（用于分隔同一输入中的两个不同句子）
- `pad_token`: 填充令牌（用于使令牌数组大小相同以便批处理）
- `cls_token`: 分类令牌（表示输入的类别）
- `mask_token`: 掩码令牌（用于掩码语言建模预训练目标）
- `additional_special_tokens`: 额外的特殊令牌列表

---

## 初始化和配置方法

### init

**功能**: 初始化 SpecialTokensMixin 实例，设置特殊令牌相关的基础配置。

**参数**:
- `verbose` (`bool`, 可选, 默认为False): 是否显示详细信息
- `**kwargs`: 各种特殊令牌的配置参数，支持的键包括：
  - `bos_token`: 句子开始令牌
  - `eos_token`: 句子结束令牌
  - `unk_token`: 未知令牌
  - `sep_token`: 分隔令牌
  - `pad_token`: 填充令牌
  - `cls_token`: 分类令牌
  - `mask_token`: 掩码令牌
  - `additional_special_tokens`: 额外特殊令牌列表或元组

**说明**:
- 初始化填充令牌类型ID为0
- 创建特殊令牌映射字典
- 允许直接设置尚不在词汇表中的特殊令牌
- 支持字符串和 AddedToken 类型的令牌值

---

## 特殊令牌管理方法

### add_special_tokens

**功能**: 向编码器添加特殊令牌字典（eos, pad, cls等），并将它们链接到类属性。

**参数**:
- `special_tokens_dict` (`dict[str, Union[str, AddedToken, Sequence[Union[str, AddedToken]]]]`): 特殊令牌字典
  - 键应该是预定义特殊属性列表中的值：`[bos_token, eos_token, unk_token, sep_token, pad_token, cls_token, mask_token, additional_special_tokens]`
  - 只有不在词汇表中的令牌才会被添加（通过检查tokenizer是否为其分配了unk_token的索引来判断）
- `replace_additional_special_tokens` (`bool`, 可选, 默认为True):
  - 如果为True，现有的额外特殊令牌列表将被 `special_tokens_dict` 中提供的列表替换
  - 如果为False，`self._special_tokens_map["additional_special_tokens"]` 只是被扩展

**返回值**:
- `int`: 添加到词汇表中的令牌数量

**示例**:
```python
# 让我们看看如何向GPT-2添加新的分类令牌
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2Model.from_pretrained("openai-community/gpt2")

special_tokens_dict = {"cls_token": "<CLS>"}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print("We have added", num_added_toks, "tokens")
# 注意：resize_token_embeddings期望接收新词汇表的全尺寸，即分词器的长度
model.resize_token_embeddings(len(tokenizer))

assert tokenizer.cls_token == "<CLS>"
```

**说明**:
- 当向词汇表添加新令牌时，应该确保调整模型的令牌嵌入矩阵大小
- 使用 `add_special_tokens` 确保特殊令牌可以多种方式使用：
  - 使用 `skip_special_tokens = True` 解码时可以跳过特殊令牌
  - 特殊令牌被tokenizer仔细处理（它们从不被分割），类似于 `AddedTokens`
  - 可以使用如 `tokenizer.cls_token` 等tokenizer类属性轻松引用特殊令牌

### sanitize_special_tokens

**功能**: 清理特殊令牌，保持向后兼容性（已弃用）。

**返回值**:
- `int`: 添加的令牌数量

**说明**:
- 此方法在transformers v5中将被移除
- 现在仅用于向后兼容
- 内部调用 `add_tokens(self.all_special_tokens_extended, special_tokens=True)`

---

## 令牌添加方法

### add_tokens

**功能**: 向tokenizer类添加新令牌列表。

**参数**:
- `new_tokens` (`str`, `tokenizers.AddedToken` 或 `str`/`tokenizers.AddedToken` 序列):
  - 只有不在词汇表中的令牌才会被添加
  - `tokenizers.AddedToken` 包装字符串令牌，让你个性化其行为：是否只匹配单个单词，是否去除左侧所有潜在空白，是否去除右侧所有潜在空白等
- `special_tokens` (`bool`, 可选, 默认为False):
  - 可用于指定令牌是否为特殊令牌
  - 这主要改变规范化行为（特殊令牌如CLS通常不大写等）

**返回值**:
- `int`: 添加到词汇表中的令牌数量

**示例**:
```python
# 让我们看看如何增加Bert模型和tokenizer的词汇表
tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
model = BertModel.from_pretrained("google-bert/bert-base-uncased")

num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
print("We have added", num_added_toks, "tokens")
# 注意：resize_token_embeddings期望接收新词汇表的全尺寸，即分词器的长度
model.resize_token_embeddings(len(tokenizer))
```

**说明**:
- 如果新令牌不在词汇表中，它们会被添加到词汇表中，索引从当前词汇表长度开始
- 添加的令牌和来自tokenization算法词汇表的令牌因此不会以相同方式处理
- 添加新令牌到词汇表时，应该确保调整模型的令牌嵌入矩阵大小

### _add_tokens

**功能**: 内部方法，执行实际的令牌添加逻辑。

**参数**:
- `new_tokens` (`Union[list[str], list[AddedToken]]`): 要添加的令牌列表
- `special_tokens` (`bool`): 是否为特殊令牌

**返回值**:
- `int`: 添加的令牌数量

**说明**: 这是一个抽象方法，需要在子类中实现具体的令牌添加逻辑。

---

## 属性访问和设置方法

### `__setattr__(self, key, value)`

**功能**: 重写属性设置方法，处理特殊令牌属性的自动转换。

**参数**:
- `key`: 属性键名
- `value`: 属性值

**说明**:
- 处理以 `_id` 或 `_ids` 结尾的键名
- 如果设置的是特殊令牌ID值，会自动转换为对应的令牌字符串
- 确保特殊令牌映射的一致性
- 对非字符串值进行验证

### `__getattr__(self, key)`

**功能**: 重写属性获取方法，提供特殊令牌的灵活访问。

**参数**:
- `key`: 属性键名

**返回值**:
- 属性值，可能是字符串或令牌ID

**说明**:
- 支持直接访问特殊令牌（如 `tokenizer.bos_token`）
- 支持访问特殊令牌ID（如 `tokenizer.bos_token_id`）
- 自动处理令牌和令牌ID之间的转换
- 如果特殊令牌未设置，返回None并记录错误

---

## 查询和获取方法

### special_tokens_map

**功能**: 获取特殊令牌映射字典。

**返回值**:
- `dict[str, Union[str, list[str]]]`: 将特殊令牌类属性（`cls_token`, `unk_token`等）映射到其值（`'<unk>'`, `'<cls>'`等）的字典

**说明**:
- 将 `tokenizers.AddedToken` 类型的潜在令牌转换为字符串
- 只包含已设置的非空特殊令牌

### `special_tokens_map_extended` (property)

**功能**: 获取扩展的特殊令牌映射字典。

**返回值**:
- `dict[str, Union[str, tokenizers.AddedToken, list[Union[str, tokenizers.AddedToken]]]]`: 特殊令牌映射字典

**说明**:
- 不将 `tokenizers.AddedToken` 类型的令牌转换为字符串
- 允许更精细地控制特殊令牌的tokenization方式
- 保持原始的令牌对象类型

### `all_special_tokens_extended` (property)

**功能**: 获取所有特殊令牌的扩展列表。

**返回值**:
- `list[Union[str, tokenizers.AddedToken]]`: 所有特殊令牌（`'<unk>'`, `'<cls>'`等）的列表

**说明**:
- 顺序与每个令牌的索引无关
- 如果想知道正确的索引，请检查 `self.added_tokens_encoder`
- 不能创建顺序，因为键是 `AddedTokens` 而不是 `Strings`
- 不将 `tokenizers.AddedToken` 类型的令牌转换为字符串
- 自动去除重复令牌

### `all_special_tokens` (property)

**功能**: 获取所有特殊令牌的字符串列表。

**返回值**:
- `list[str]`: 唯一特殊令牌（`'<unk>'`, `'<cls>'`...等）的列表

**说明**:
- 将 `tokenizers.AddedToken` 类型的令牌转换为字符串
- 基于扩展令牌列表生成字符串版本

### `all_special_ids` (property)

**功能**: 获取所有特殊令牌ID的列表。

**返回值**:
- `list[int]`: 映射到类属性的特殊令牌（`'<unk>'`, `'<cls>'`等）的ID列表

**说明**:
- 获取所有特殊令牌字符串
- 将这些字符串转换为对应的令牌ID
- 返回ID列表

### `pad_token_type_id` (property)

**功能**: 获取填充令牌类型的ID。

**返回值**:
- `int`: 词汇表中填充令牌类型的ID

**说明**:
- 返回内部存储的填充令牌类型ID
- 默认值为0

---

## 内部和扩展方法

### `_set_model_specific_special_tokens(self, special_tokens: list[str])`

**功能**: 向 "SPECIAL_TOKENS_ATTRIBUTES" 列表添加新的特殊令牌，使其成为 "self.special_tokens" 的一部分并在tokenizer配置中保存为特殊令牌。

**参数**:
- `special_tokens` (`list[str]`): 特殊令牌字典，键为令牌名称，值为令牌值

**说明**:
- 允许在初始化tokenizer后动态添加新的模型特定令牌
- 例如：如果模型tokenizer是多模态的，可以支持特殊的图像或音频令牌
- 将新令牌添加到特殊令牌属性列表
- 验证令牌类型必须是字符串或AddedToken

---

## 使用示例

### 基本使用

```python
# 初始化tokenizer时设置特殊令牌
tokenizer = MyTokenizer(
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    cls_token="<cls>",
    sep_token="<sep>"
)

# 访问特殊令牌
print(tokenizer.bos_token)  # "<s>"
print(tokenizer.bos_token_id)  # 对应的ID

# 获取所有特殊令牌
all_special = tokenizer.all_special_tokens
all_special_ids = tokenizer.all_special_ids
```

### 添加特殊令牌

```python
# 添加新的特殊令牌
special_tokens = {
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "additional_special_tokens": ["[ENT]", "[REL]"]
}

num_added = tokenizer.add_special_tokens(special_tokens)
print(f"添加了 {num_added} 个特殊令牌")
```

### 动态扩展特殊令牌

```python
# 为多模态模型添加图像和音频特殊令牌
model_specific_tokens = {
    "image_token": "<image>",
    "audio_token": "<audio>"
}

tokenizer._set_model_specific_special_tokens(model_specific_tokens)
```

---

## 总结

`SpecialTokensMixin` 类提供了完整的特殊令牌管理框架：

1. **统一的接口**: 以模型无关的方式处理特殊令牌
2. **灵活的配置**: 支持所有常见的特殊令牌类型
3. **动态管理**: 允许运行时添加和修改特殊令牌
4. **自动转换**: 智能处理令牌字符串和ID之间的转换
5. **扩展性**: 支持模型特定的自定义特殊令牌
6. **向后兼容**: 保持API的稳定性和一致性

这个混入类为所有Transformers分词器提供了统一而强大的特殊令牌处理能力，使得不同模型的特殊令牌管理具有一致的API和行为。