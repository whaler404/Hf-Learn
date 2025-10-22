# PreTrainedTokenizerFast 方法文档

`PreTrainedTokenizerFast` 是 Hugging Face Transformers 库中快速分词器的基类，它包装了 Rust 实现的 tokenizers 库以提供高性能的分词功能，继承自 `PreTrainedTokenizerBase`

## 目录
- [初始化和配置方法](#初始化和配置方法)
- [词汇表相关方法](#词汇表相关方法)
- [编码和解码方法](#编码和解码方法)
- [特殊token处理方法](#特殊token处理方法)
- [训练和保存方法](#训练和保存方法)
- [内部工具方法](#内部工具方法)

---

## 初始化和配置方法

### `__init__(self, *args, **kwargs)`

初始化快速分词器实例。

**主要参数：**
- `tokenizer_object`: 来自 🤗 tokenizers 的 `Tokenizer` 对象
- `tokenizer_file`: 本地 JSON 文件路径，表示序列化的 `Tokenizer` 对象
- `from_slow`: 是否从慢速分词器转换，默认 `False`
- `__slow_tokenizer`: 慢速分词器实例
- `gguf_file`: GGUF 格式文件路径
- `add_prefix_space`: 是否添加前缀空格，默认 `False`

**支持多种初始化方式：**
1. 从现有的 tokenizer 对象创建
2. 从序列化的 tokenizer 文件加载
3. 从慢速分词器转换
4. 从 GGUF 文件创建
5. 使用默认的慢速分词器类创建

---

## 词汇表相关方法

### vocab_size (属性)

```python
@property
def vocab_size(self) -> int
```

**功能：** 返回基础词汇表的大小（不包括添加的 tokens）

**返回值：**
- `int`: 基础词汇表大小

### get_vocab()

```python
def get_vocab(self) -> dict[str, int]
```

**功能：** 获取包含添加 tokens 的完整词汇表

**返回值：**
- `dict[str, int]`: 词汇到索引的映射字典

### vocab (属性)

```python
@property
def vocab(self) -> dict[str, int]
```

**功能：** 获取词汇表，等同于 `get_vocab()`

**返回值：**
- `dict[str, int]`: 词汇到索引的映射字典

### added_tokens_encoder (属性)

```python
@property
def added_tokens_encoder(self) -> dict[str, int]
```

**功能：** 返回从字符串到索引的排序映射，包含添加的 tokens

**返回值：**
- `dict[str, int]`: 添加的 token 到索引的映射

### added_tokens_decoder (属性)

```python
@property
def added_tokens_decoder(self) -> dict[int, AddedToken]
```

**功能：** 返回词汇表中添加的 tokens，格式为索引到 AddedToken 的字典

**返回值：**
- `dict[int, AddedToken]`: 索引到 AddedToken 的映射

### get_added_vocab()

```python
def get_added_vocab(self) -> dict[str, int]
```

**功能：** 返回添加的 tokens，格式为 token 到索引的字典

**返回值：**
- `dict[str, int]`: 添加的 token 到索引的映射

---

## 编码和解码方法

### convert_tokens_to_ids

```python
def convert_tokens_to_ids(self, tokens: Union[str, Iterable[str]]) -> Union[int, list[int]]
```

**功能：** 将 token 字符串（或 token 序列）转换为整数 id（或 id 列表）

**参数：**
- `tokens` (`str` 或 `Iterable[str]`): 要转换的一个或多个 tokens

**返回值：**
- `int` 或 `list[int]`: token id 或 token id 列表

### convert_ids_to_tokens

```python
def convert_ids_to_tokens(
    self, ids: Union[int, list[int]], skip_special_tokens: bool = False
) -> Union[str, list[str]]
```

**功能：** 将单个索引或索引序列转换为 token 或 token 序列

**参数：**
- `ids` (`int` 或 `list[int]`): 要转换的 token id 或 token id 列表
- `skip_special_tokens` (`bool`, 可选, 默认为 `False`): 是否在解码时移除特殊 tokens

**返回值：**
- `str` 或 `list[str]`: 解码后的 token(s)

### tokenize()

```python
def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> list[str]
```

**功能：** 对文本进行分词

**参数：**
- `text` (`str`): 要分词的文本
- `pair` (`str`, 可选): 文本对中的第二个文本
- `add_special_tokens` (`bool`, 默认为 `False`): 是否添加特殊 tokens

**返回值：**
- `list[str]`: 分词后的 token 列表

### convert_tokens_to_string

```python
def convert_tokens_to_string(self, tokens: list[str]) -> str
```

**功能：** 将 token 列表转换为字符串

**参数：**
- `tokens` (`list[str]`): 要转换的 token 列表

**返回值：**
- `str`: 解码后的字符串

### _decode

```python
def _decode(
    self,
    token_ids: Union[int, list[int]],
    skip_special_tokens: bool = False,
    clean_up_tokenization_spaces: Optional[bool] = None,
    **kwargs,
) -> str
```

**功能：** 内部解码方法，将 token id(s) 转换为字符串

**参数：**
- `token_ids` (`int` 或 `list[int]`): 要解码的 token id(s)
- `skip_special_tokens` (`bool`, 可选, 默认为 `False`): 是否跳过特殊 tokens
- `clean_up_tokenization_spaces` (`bool`, 可选): 是否清理分词空格

**返回值：**
- `str`: 解码后的文本

---

## 特殊token处理方法

### num_special_tokens_to_add

```python
def num_special_tokens_to_add(self, pair: bool = False) -> int
```

**功能：** 返回编码序列时添加的特殊 tokens 数量

⚠️ **注意：** 此方法通过编码虚拟输入并检查添加的 tokens 数量来实现，因此效率不高。不要在训练循环中使用。

**参数：**
- `pair` (`bool`, 可选, 默认为 `False`): 是否计算序列对情况下的特殊 tokens 数量

**返回值：**
- `int`: 添加到序列的特殊 tokens 数量

### set_truncation_and_padding

```python
def set_truncation_and_padding(
    self,
    padding_strategy: PaddingStrategy,
    truncation_strategy: TruncationStrategy,
    max_length: int,
    stride: int,
    pad_to_multiple_of: Optional[int],
    padding_side: Optional[str],
)
```

**功能：** 为快速分词器定义截断和填充策略，并在之后恢复分词器设置

**参数：**
- `padding_strategy` (`PaddingStrategy`): 应用于输入的填充类型
- `truncation_strategy` (`TruncationStrategy`): 应用于输入的截断类型
- `max_length` (`int`): 序列的最大长度
- `stride` (`int`): 处理溢出时的步长
- `pad_to_multiple_of` (`int`, 可选): 如果设置，将序列填充到指定值的倍数
- `padding_side` (`str`, 可选): 应用填充的一侧，可选值为 ['right', 'left']

---

## 训练和保存方法

### train_new_from_iterator

```python
def train_new_from_iterator(
    self,
    text_iterator,
    vocab_size,
    length=None,
    new_special_tokens=None,
    special_tokens_map=None,
    **kwargs,
)
```

**功能：** 在新语料库上训练分词器，使用与当前分词器相同的默认设置（特殊 tokens 或分词流程）

**参数：**
- `text_iterator` (`generator of list[str]`): 训练语料库，应该是文本批次的生成器
- `vocab_size` (`int`): 期望的分词器词汇表大小
- `length` (`int`, 可选): 迭代器中的序列总数，用于提供有意义的进度跟踪
- `new_special_tokens` (`list of str or AddedToken`, 可选): 要添加到训练分词器的新特殊 tokens 列表
- `special_tokens_map` (`dict[str, str]`, 可选): 如果要重命名的特殊 tokens，提供旧名称到新名称的映射
- `**kwargs` (`dict[str, Any]`, 可选): 传递给 🤗 Tokenizers 库训练器的额外关键字参数

**返回值：**
- `PreTrainedTokenizerFast`: 与原始类型相同的新分词器，在 `text_iterator` 上训练

### _save_pretrained

```python
def _save_pretrained(
    self,
    save_directory: Union[str, os.PathLike],
    file_names: tuple[str, ...],
    legacy_format: Optional[bool] = None,
    filename_prefix: Optional[str] = None,
) -> tuple[str, ...]
```

**功能：** 使用慢速分词器/传统格式保存分词器：词汇表 + 添加的 tokens，以及包含 {配置 + 词汇表 + 添加的 tokens} 的唯一 JSON 文件

**参数：**
- `save_directory` (`str` 或 `os.PathLike`): 保存目录
- `file_names` (`tuple[str, ...]`): 文件名元组
- `legacy_format` (`bool`, 可选): 是否使用传统格式
- `filename_prefix` (`str`, 可选): 文件名前缀

**返回值：**
- `tuple[str, ...]`: 保存的文件名元组

---

## 内部工具方法

### `is_fast` (属性)

```python
@property
def is_fast(self) -> bool
```

**功能：** 返回 True，标识这是快速分词器

**返回值：**
- `bool`: 始终为 True

### `can_save_slow_tokenizer` (属性)

```python
@property
def can_save_slow_tokenizer(self) -> bool
```

**功能：** 返回是否可以保存慢速分词器

**返回值：**
- `bool`: 如果可以保存慢速分词器则返回 True

### `__bool__()`

```python
def __bool__(self) -> bool
```

**功能：** 返回 True，避免昂贵的 `assert tokenizer` 陷阱

**返回值：**
- `bool`: 始终为 True

### `__len__()`

```python
def __len__(self) -> int
```

**功能：** 返回包含添加 tokens 的完整词汇表大小

**返回值：**
- `int`: 完整词汇表大小

### `backend_tokenizer` (属性)

```python
@property
def backend_tokenizer(self) -> TokenizerFast
```

**功能：** 返回用作后端的 Rust 分词器

**返回值：**
- `TokenizerFast`: Rust 后端分词器

### `decoder` (属性)

```python
@property
def decoder(self) -> DecoderFast
```

**功能：** 返回此分词器的 Rust 解码器

**返回值：**
- `DecoderFast`: Rust 解码器

### `_convert_encoding()`

```python
def _convert_encoding(
    self,
    encoding: EncodingFast,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
) -> tuple[dict[str, Any], list[EncodingFast]]
```

**功能：** 将编码表示（来自底层 HuggingFace 分词器输出）转换为 Python 字典和编码列表，处理来自溢出 tokens 的批次构建

**参数：**
- `encoding` (`EncodingFast`): 输入编码
- `return_token_type_ids` (`bool`, 可选): 是否返回 token 类型 ids
- `return_attention_mask` (`bool`, 可选): 是否返回注意力掩码
- `return_overflowing_tokens` (`bool`, 可选): 是否返回溢出 tokens
- `return_special_tokens_mask` (`bool`, 可选): 是否返回特殊 tokens 掩码
- `return_offsets_mapping` (`bool`, 可选): 是否返回偏移映射
- `return_length` (`bool`, 可选): 是否返回长度
- `verbose` (`bool`, 可选): 是否显示详细信息

**返回值：**
- `tuple[dict[str, Any], list[EncodingFast]]`: 编码字典和编码列表的元组

### `_convert_token_to_id_with_added_voc()`

```python
def _convert_token_to_id_with_added_voc(self, token: str) -> int
```

**功能：** 将 token 转换为 id，考虑添加的词汇表

**参数：**
- `token` (`str`): 要转换的 token

**返回值：**
- `int`: token id 或 unknown token id

### `_convert_id_to_token()`

```python
def _convert_id_to_token(self, index: int) -> Optional[str]
```

**功能：** 将 id 转换为 token

**参数：**
- `index` (`int`): 要转换的 id

**返回值：**
- `Optional[str]`: 对应的 token，如果不存在则返回 None

### `_add_tokens()`

```python
def _add_tokens(self, new_tokens: list[Union[str, AddedToken]], special_tokens=False) -> int
```

**功能：** 添加新的 tokens 到词汇表

**参数：**
- `new_tokens` (`list[Union[str, AddedToken]]`): 要添加的新 tokens 列表
- `special_tokens` (`bool`, 可选): 是否为特殊 tokens

**返回值：**
- `int`: 添加的 tokens 数量

### `_batch_encode_plus()`

```python
def _batch_encode_plus(
    self,
    batch_text_or_text_pairs: Union[
        list[TextInput], list[TextInputPair], list[PreTokenizedInput], list[PreTokenizedInputPair]
    ],
    add_special_tokens: bool = True,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    max_length: Optional[int] = None,
    stride: int = 0,
    is_split_into_words: bool = False,
    pad_to_multiple_of: Optional[int] = None,
    padding_side: Optional[str] = None,
    return_tensors: Optional[str] = None,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
    split_special_tokens: bool = False,
) -> BatchEncoding
```

**功能：** 批量编码文本或文本对

**参数：**
- `batch_text_or_text_pairs`: 要编码的文本批次
- `add_special_tokens`: 是否添加特殊 tokens
- `padding_strategy`: 填充策略
- `truncation_strategy`: 截断策略
- `max_length`: 最大长度
- `stride`: 步长
- `is_split_into_words`: 是否已经分割为词语
- `pad_to_multiple_of`: 填充到指定值的倍数
- `padding_side`: 填充侧
- `return_tensors`: 返回张量类型
- 其他可选返回参数...

**返回值：**
- `BatchEncoding`: 编码结果

### `_encode_plus()`

```python
def _encode_plus(
    self,
    text: Union[TextInput, PreTokenizedInput],
    text_pair: Optional[Union[TextInput, PreTokenizedInput]] = None,
    add_special_tokens: bool = True,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    max_length: Optional[int] = None,
    stride: int = 0,
    is_split_into_words: bool = False,
    pad_to_multiple_of: Optional[int] = None,
    padding_side: Optional[str] = None,
    return_tensors: Optional[bool] = None,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
    split_special_tokens: bool = False,
    **kwargs,
) -> BatchEncoding
```

**功能：** 编码单个文本或文本对

**参数：** 类似 `_batch_encode_plus()` 但针对单个文本

**返回值：**
- `BatchEncoding`: 编码结果

---

## 使用示例

### 基本使用

```python
from transformers import AutoTokenizer

# 加载快速分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 编码文本
text = "Hello, world!"
encoded = tokenizer(text)
print(encoded["input_ids"])  # [101, 7592, 1010, 2088, 999, 102]

# 解码文本
decoded = tokenizer.decode(encoded["input_ids"])
print(decoded)  # [CLS] hello, world! [SEP]

# 转换 tokens 和 ids
tokens = tokenizer.tokenize(text)
print(tokens)  # ['hello', ',', 'world', '!']

token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)  # [7592, 1010, 2088, 999]
```

### 批量编码

```python
# 批量编码
texts = ["Hello world", "How are you?"]
encoded_batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
print(encoded_batch["input_ids"].shape)  # torch.Size([2, 7])
```

### 训练新分词器

```python
# 使用现有分词器训练新分词器
corpus = [
    ["This is the first sentence.", "This is the second sentence."],
    ["Another example sentence.", "And one more sentence."]
]

new_tokenizer = tokenizer.train_new_from_iterator(
    corpus,
    vocab_size=30000,
    length=len(corpus)
)
```

---

## 性能注意事项

1. **快速 vs 慢速**: `PreTrainedTokenizerFast` 使用 Rust 后端，比纯 Python 实现的慢速分词器快得多
2. **内存效率**: 快速分词器在处理大量文本时内存使用更高效
3. **特殊 tokens**: 使用 `num_special_tokens_to_add()` 时要注意性能影响
4. **批处理**: 尽量使用批量编码而不是单个编码以提高性能