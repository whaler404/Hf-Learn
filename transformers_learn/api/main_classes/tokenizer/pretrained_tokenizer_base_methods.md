# PreTrainedTokenizerBase 方法文档

本文档整理了 `PreTrainedTokenizerBase` 类提供的所有方法，按功能分类说明。

```python
class PaddingStrategy(ExplicitEnum):
    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"

class TruncationStrategy(ExplicitEnum):
    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"
```

## 目录

- [初始化和配置方法](#初始化和配置方法)
- [词汇表和特殊令牌管理](#词汇表和特殊令牌管理)
- [模型加载和保存](#模型加载和保存)
- [文本编码方法](#文本编码方法)
- [文本解码方法](#文本解码方法)
- [批处理方法](#批处理方法)
- [聊天模板方法](#聊天模板方法)
- [工具方法](#工具方法)
- [特殊上下文管理](#特殊上下文管理)

---

## 初始化和配置方法

### init

`__init__(self, **kwargs)`

**功能**: 初始化分词器基类，设置基本配置参数。

**参数**:
- `**kwargs`: 各种配置参数，包括：
  - `name_or_path`: 模型名称或路径
  - `model_max_length`: 模型最大长度
  - `padding_side`: 填充方向 ('right' 或 'left')
  - `truncation_side`: 截断方向 ('right' 或 'left')
  - `clean_up_tokenization_spaces`: 是否清理分词空格
  - `split_special_tokens`: 是否分割特殊令牌
  - `chat_template`: 聊天模板
  - `model_input_names`: 模型输入名称列表

**说明**: 这是基类的构造函数，处理所有分词器共享的初始化逻辑。验证配置参数的有效性，设置默认值，并初始化各种属性。

---

## 词汇表和特殊令牌管理

### get_vocab

`get_vocab(self) -> dict[str, int]`

**功能**: 获取词汇表，返回令牌到索引的映射字典。

**返回值**:
- `dict[str, int]`: 词汇表字典，键为令牌字符串，值为对应的整数索引

**说明**:
- `tokenizer.get_vocab()[token]` 等同于 `tokenizer.convert_tokens_to_ids(token)`（当令牌在词汇表中时）
- 这是一个抽象方法，需要在子类中实现

### len

**功能**: 返回词汇表大小。

**返回值**:
- `int`: 词汇表中的令牌数量

**说明**: 这是一个抽象方法，需要在子类中实现。

### added_tokens_decoder

**功能**: 返回添加的令牌解码器。

**返回值**:
- `dict[int, AddedToken]`: 令牌ID到AddedToken对象的映射

**说明**: 这是一个抽象方法，需要在子类中实现。

---

## 模型加载和保存

### from_pretrained

**功能**: 从预定义的tokenizer实例化PreTrainedTokenizerBase或其派生类。

**参数**:
- `pretrained_model_name_or_path` (`str` 或 `os.PathLike`): 可以是：
  - HuggingFace上托管的预定义tokenizer的模型ID字符串
  - 包含tokenizer所需词汇文件的目录路径
  - 单个保存的词汇文件的路径或URL（已弃用）
- `cache_dir` (`str` 或 `os.PathLike`, 可选): 下载的预定义tokenizer词汇文件的缓存目录
- `force_download` (`bool`, 可选, 默认为False): 是否强制重新下载词汇文件并覆盖缓存版本
- `local_files_only` (`bool`, 可选, 默认为False): 是否仅依赖本地文件，不尝试下载任何文件
- `token` (`str` 或 `bool`, 可选): 用作HTTP bearer授权的令牌，用于远程文件
- `revision` (`str`, 可选, 默认为"main"): 要使用的特定模型版本
- `trust_remote_code` (`bool`, 可选, 默认为False): 是否允许Hub上定义的自定义模型
- `**kwargs`: 传递给Tokenizer初始化方法的关键字参数

**返回值**:
- 实例化的tokenizer对象

**示例**:
```python
# 从HuggingFace下载并缓存词汇表
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

# 从目录加载（如果tokenizer使用save_pretrained保存）
tokenizer = BertTokenizer.from_pretrained("./test/saved_model/")

# 如果tokenizer使用单个词汇文件，可以直接指向该文件
tokenizer = BertTokenizer.from_pretrained("./test/saved_model/my_vocab.txt")

# 在实例化时链接特殊词汇
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", unk_token="<unk>")
```

### save_pretrained

`save_pretrained(self, save_directory, legacy_format=None, filename_prefix=None)`

**功能**: 保存tokenizer词汇文件和配置到指定目录。

**参数**:
- `save_directory` (`str` 或 `os.PathLike`): 保存tokenizer文件的目录路径
- `legacy_format` (`bool`, 可选): 是否使用旧格式保存
- `filename_prefix` (`str`, 可选): 保存文件的前缀

**说明**: 保存tokenizer的所有必要文件，包括词汇表、配置文件等，以便后续可以使用from_pretrained重新加载。

### save_vocabulary

**功能**: 保存词汇表文件到指定目录。

**参数**:
- `save_directory` (`str`): 保存目录路径
- `filename_prefix` (`str`, 可选): 文件名前缀

**返回值**:
- `tuple[str, ...]`: 保存的文件路径元组

**说明**: 这是一个抽象方法，需要在子类中实现具体的词汇表保存逻辑。

---

## 文本编码方法

### tokenize

**功能**: 将字符串转换为令牌序列，用unk_token替换未知令牌。

**参数**:
- `text` (`str`): 要编码的序列
- `pair` (`str`, 可选): 要与第一个序列一起编码的第二个序列
- `add_special_tokens` (`bool`, 可选, 默认为False): 是否添加相应模型关联的特殊令牌
- `**kwargs`: 传递给底层模型特定encode方法的额外关键字参数

**返回值**:
- `list[str]`: 令牌列表

**说明**: 这是一个抽象方法，需要在子类中实现具体的分词逻辑。

### encode

**功能**: 使用tokenizer和词汇表将字符串转换为id序列（整数）。

**参数**:
- `text` (`str`, `list[str]` 或 `list[int]`): 要编码的第一个序列，可以是字符串、令牌列表或ID列表
- `text_pair` (`str`, `list[str]` 或 `list[int]`, 可选): 要编码的第二个序列
- `add_special_tokens` (`bool`, 可选, 默认为True): 是否添加特殊令牌
- `padding` (`bool`, `str` 或 `PaddingStrategy`, 可选): 填充策略
- `truncation` (`bool`, `str` 或 `TruncationStrategy`, 可选): 截断策略
- `max_length` (`int`, 可选): 填充或截断的最大长度
- `stride` (`int`, 可选): 截断时的步长
- `padding_side` (`str`, 可选): 填充方向
- `return_tensors` (`str` 或 `TensorType`, 可选): 返回张量的框架类型

**返回值**:
- `list[int]`: 编码后的令牌ID列表

**说明**:
- 等同于执行 `self.convert_tokens_to_ids(self.tokenize(text))`
- 内部调用encode_plus方法并返回input_ids

### call

**功能**: 对一个或多个序列或序列对进行分词并准备模型输入的主要方法。

**参数**:
- `text` (`str`, `list[str]`, `list[list[str]]`, 可选): 要编码的序列或序列批次
- `text_pair` (`str`, `list[str]`, `list[list[str]]`, 可选): 要编码的第二序列或序列批次
- `text_target` (`str`, `list[str]`, `list[list[str]]`, 可选): 要编码为目标文本的序列
- `text_pair_target` (`str`, `list[str]`, `list[list[str]]`, 可选): 要编码为目标文本的第二序列
- `add_special_tokens` (`bool`, 可选, 默认为True): 是否添加特殊令牌
- `padding` (`bool`, `str` 或 `PaddingStrategy`, 可选): 填充策略
- `truncation` (`bool`, `str` 或 `TruncationStrategy`, 可选): 截断策略
- `max_length` (`int`, 可选): 最大长度
- `stride` (`int`, 可选): 截断步长
- `is_split_into_words` (`bool`, 可选): 输入是否已预分词
- `pad_to_multiple_of` (`int`, 可选): 填充到指定倍数
- `padding_side` (`str`, 可选): 填充方向
- `return_tensors` (`str` 或 `TensorType`, 可选): 返回张量类型
- `return_token_type_ids` (`bool`, 可选): 是否返回令牌类型ID
- `return_attention_mask` (`bool`, 可选): 是否返回注意力掩码
- `return_overflowing_tokens` (`bool`, 可选): 是否返回溢出令牌
- `return_special_tokens_mask` (`bool`, 可选): 是否返回特殊令牌掩码
- `return_offsets_mapping` (`bool`, 可选): 是否返回偏移映射
- `return_length` (`bool`, 可选): 是否返回长度信息
- `verbose` (`bool`, 可选): 是否显示详细信息

**返回值**:
- `BatchEncoding`: 包含所有编码结果的批次对象

**说明**:
- 这是最主要的编码方法，统一处理单个或批量文本的编码
- 支持源文本和目标文本的同时编码（用于序列到序列任务）
- 提供丰富的输出选项以适应不同的模型需求

---

## 文本解码方法

### decode

**功能**: 使用tokenizer和词汇表将ID序列转换为字符串，可选择移除特殊令牌和清理分词空格。

**参数**:
- `token_ids` (`Union[int, list[int], np.ndarray, torch.Tensor, tf.Tensor]`): 令牌化输入ID列表，可通过`__call__`方法获得
- `skip_special_tokens` (`bool`, 可选, 默认为False): 是否在解码中移除特殊令牌
- `clean_up_tokenization_spaces` (`bool`, 可选): 是否清理分词空格，如果为None则使用默认值
- `**kwargs`: 传递给底层模型特定decode方法的额外关键字参数

**返回值**:
- `str`: 解码后的句子

**说明**:
- 等同于执行 `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`
- 内部调用_decode抽象方法

### batch_decode

**功能**: 通过调用decode将令牌ID列表列表转换为字符串列表。

**参数**:
- `sequences` (`Union[list[int], list[list[int]], np.ndarray, torch.Tensor, tf.Tensor]`): 令牌化输入ID列表，可通过`__call__`方法获得
- `skip_special_tokens` (`bool`, 可选, 默认为False): 是否在解码中移除特殊令牌
- `clean_up_tokenization_spaces` (`bool`, 可选): 是否清理分词空格，如果为None则使用默认值
- `**kwargs`: 传递给底层模型特定decode方法的额外关键字参数

**返回值**:
- `list[str]`: 解码后的句子列表

**说明**: 对输入的每个序列调用decode方法，适用于批量解码场景。

### _decode

**功能**: 内部解码方法，执行实际的解码逻辑。

**参数**:
- `token_ids` (`Union[int, list[int]]`): 要解码的令牌ID
- `skip_special_tokens` (`bool`, 可选, 默认为False): 是否跳过特殊令牌
- `clean_up_tokenization_spaces` (`bool`, 可选): 是否清理分词空格
- `**kwargs`: 额外的关键字参数

**返回值**:
- `str`: 解码后的字符串

**说明**: 这是一个抽象方法，需要在子类中实现具体的解码逻辑。

---

## 批处理方法

### batch_encode_plus

**功能**: 批量编码文本序列对的主要方法。

**参数**:
- `batch_text_or_text_pairs`: 文本或文本对的批次
- 其他参数与`__call__`方法类似

**返回值**:
- `BatchEncoding`: 包含批量编码结果的对象

**说明**: 专门用于批量处理文本序列，提供与单个文本编码相同的功能但针对批次优化。

### encode_plus

**功能**: 编码单个文本序列对并返回完整的编码信息。

**参数**:
- `text`: 要编码的第一个序列
- `text_pair`: 要编码的第二个序列（可选）
- 其他参数与`__call__`方法类似

**返回值**:
- `BatchEncoding`: 包含所有编码信息的对象

**说明**: 提供比encode方法更详细的输出，包括注意力掩码、令牌类型ID等信息。

### pad

**功能**: 对编码输入进行填充以使其长度一致。

**参数**:
- `encoded_inputs` （`Union[BatchEncoding, list[BatchEncoding]]`）: 要填充的编码输入
- `padding` （`Union[bool, str, PaddingStrategy]=True`）: 填充策略
- `max_length`: 最大长度
- `pad_to_multiple_of`: 填充到指定倍数
- `padding_side`: 填充方向
- `return_attention_mask`: 是否返回注意力掩码
- `return_tensors`: 返回张量类型
- `verbose`: 是否显示详细信息

**返回值**:
- `BatchEncoding`: 填充后的编码结果

**说明**: 用于批量处理时统一序列长度，确保所有序列具有相同的长度以便模型处理。

---

## 聊天模板方法

### apply_chat_template

**功能**: 将包含"role"和"content"键的字典列表转换为令牌ID列表，用于聊天模型。

**参数**:
- `conversation`: 包含"role"和"content"键的字典列表，表示聊天历史
- `tools` (`list[Union[Dict, Callable]]`, 可选): 模型可访问的工具列表
- `documents` (`list[dict[str, str]]`, 可选): 模型可访问的文档列表（用于RAG）
- `chat_template` (`str`, 可选): 用于转换的Jinja模板
- `add_generation_prompt` (`bool`, 可选): 是否添加表示助手消息开始的令牌
- `continue_final_message` (`bool`, 可选): 是否让聊天格式化为开放式最终消息
- `tokenize` (`bool`, 可选, 默认为True): 是否对输出进行分词
- `padding`: 填充策略
- `truncation`: 是否截断序列
- `max_length`: 分词的最大长度
- `return_tensors`: 返回张量类型
- `return_dict` (`bool`, 可选): 是否返回命名输出的字典
- `return_assistant_tokens_mask` (`bool`, 可选): 是否返回助手生成令牌的掩码
- `tokenizer_kwargs` (`dict[str: Any]`, 可选): 传递给tokenizer的额外kwargs
- `**kwargs`: 传递给模板渲染器的额外kwargs

**返回值**:
- `Union[list[int], Dict]`: 表示到目前为止分词聊天的令牌ID列表，包括控制令牌

**说明**:
- 读取tokenizer的chat_template属性来确定格式和控制令牌
- 输出可直接传递给模型或通过generate()等方法使用
- 支持函数调用和RAG功能

### get_chat_template

**功能**: 获取当前tokenizer的聊天模板。

**参数**:
- `chat_template` (`str`, 可选): 要使用的Jinja模板，如果不提供则使用tokenizer的默认模板
- `tools` (`list[dict]`, 可选): 工具列表，用于模板渲染

**返回值**:
- `str`: 渲染后的聊天模板

**说明**:
- 获取并渲染聊天模板，支持工具调用功能
- 如果模板不支持工具调用，tools参数将被忽略

### encode_message_with_chat_template

**功能**: 使用聊天模板编码消息。

**参数**:
- `messages` （`dict[str, str]`）: 消息列表
- `tools`: 工具列表（可选）
- `chat_template`: 聊天模板（可选）
- `**kwargs`: 额外的渲染参数

**返回值**:
- `str`: 编码后的消息字符串

**说明**: 使用指定的聊天模板对消息进行格式化和编码。

### save_chat_templates

**功能**: 保存聊天模板到文件。

**参数**:
- `save_directory` (`str`): 保存目录
- `chat_template_vars` (`list[dict]`, 可选): 聊天模板变量列表
- `filename_prefix` (`str`, 可选): 文件名前缀

**说明**: 将当前的聊天模板配置保存到指定目录，便于后续重新加载使用。

---

## 工具方法

### convert_tokens_to_string

**功能**: 将令牌列表转换为字符串。

**参数**:
- `tokens` (`list[str]`): 令牌列表

**返回值**:
- `str`: 转换后的字符串

**说明**: 这是token_to_string转换的抽象方法，需要在子类中实现。

### `clean_up_tokenization(self, out_string: str) -> str`

**功能**: 清理分词后的字符串。

**参数**:
- `out_string` (`str`): 要清理的字符串

**返回值**:
- `str`: 清理后的字符串

**说明**: 执行分词后的清理工作，如移除多余空格、标准化格式等。

### get_special_tokens_mask

**功能**: 从无需特殊令牌的序列中创建特殊令牌掩码。

**参数**:
- `token_ids_0` (`list`): 第一个令牌ID序列
- `token_ids_1` (`list`, 可选): 第二个令牌ID序列
- `already_has_special_tokens` (`bool`, 可选): 序列是否已经包含特殊令牌

**返回值**:
- `list[int]`: 特殊令牌掩码，特殊令牌位置为1，其他为0

**说明**: 用于标识哪些位置是特殊令牌，便于在解码时选择性地跳过它们。

### create_token_type_ids_from_sequences

**功能**: 从传递的序列创建令牌类型ID。

**参数**:
- `token_ids_0` (`list`): 第一个令牌ID序列
- `token_ids_1` (`list`, 可选): 第二个令牌ID序列

**返回值**:
- `list[int]`: 令牌类型ID列表

**说明**: 为序列对创建令牌类型ID，用于区分不同序列的令牌（如句子A和句子B）。

### `build_inputs_with_special_tokens(self, token_ids_0: list, token_ids_1: Optional[list] = None) -> list[int]`

**功能**: 通过连接和添加特殊令牌从序列或序列对构建模型输入。

**参数**:
- `token_ids_0` (`list`): 要构建的令牌ID列表
- `token_ids_1` (`list`, 可选): 第二个要构建的令牌ID列表

**返回值**:
- `list[int]: 包含特殊令牌的模型输入序列

**说明**:
- 这是一个抽象方法，需要在子类中实现
- 不同模型有不同的特殊令牌添加策略（如BERT使用[CLS]和[SEP]）

### prepare_for_model

**功能**: 为模型准备输入，执行所有预处理步骤。

**参数**:
- `ids`: 主要的令牌ID列表
- `pair_ids`: 可选的第二令牌ID列表
- `add_special_tokens`: 是否添加特殊令牌
- `padding`: 填充策略
- `truncation`: 截断策略
- `max_length`: 最大长度
- `stride`: 截断步长
- `pad_to_multiple_of`: 填充倍数
- `padding_side`: 填充方向
- `return_tensors`: 返回张量类型
- `return_token_type_ids`: 是否返回令牌类型ID
- `return_attention_mask`: 是否返回注意力掩码
- `return_length`: 是否返回长度
- `verbose`: 是否显示详细信息
- `prepend_batch_axis`: 是否前置批次轴
- `return_overflowing_tokens`: 是否返回溢出令牌

**返回值**:
- `BatchEncoding`: 准备好的模型输入

**说明**: 这是最终的预处理方法，执行从原始令牌ID到模型就绪输入的所有转换步骤。

### truncate_sequences

**功能**: 使用指定策略截断序列。

**参数**:
- `ids`: 要截断的令牌ID列表
- `pair_ids`: 可选的第二令牌ID列表
- `num_tokens_to_remove`: 要移除的令牌数量
- `truncation_strategy`: 截断策略
- `stride`: 截断步长

**返回值**:
- `tuple[list[int], list[int], list[int]]`: 截断后的序列和可能的溢出令牌

**说明**: 支持多种截断策略，如从头截断、从尾截断、最长优先等。

---

## 特殊上下文管理

### as_target_tokenizer

**功能**: 上下文管理器，将tokenizer切换到目标模式。

**返回值**:
- 上下文管理器对象

**说明**:
- 在序列到序列任务中，用于区分源文本和目标文本的分词
- 目标模式下的分词可能有不同的处理方式

### _switch_to_input_mode

**功能**: 内部方法，切换到输入模式。

**说明**: 用于内部状态管理，在源文本和目标文本处理之间切换。

### _switch_to_target_mode

**功能**: 内部方法，切换到目标模式。

**说明**: 与_switch_to_input_mode配合，管理tokenizer的工作模式。

---

## 属性方法

### max_len_single_sentence

**功能**: 获取可馈送到模型的单个句子的最大长度。

**返回值**:
- `int`: 单个句子的最大长度

**说明**:
- 计算方式：`self.model_max_length - self.num_special_tokens_to_add(pair=False)`
- 对于单个序列输入的长度限制

### max_len_sentences_pair

**功能**: 获取可馈送到模型的序列对的最大组合长度。

**返回值**:
- `int`: 序列对的最大长度

**说明**:
- 计算方式：`self.model_max_length - self.num_special_tokens_to_add(pair=True)`
- 对于序列对输入的长度限制

---

## 注册和自动类支持

### register_for_auto_class

**功能**: 将此tokenizer类注册到自动类中。

**参数**:
- `auto_class` (`str`, 可选, 默认为"AutoTokenizer"): 要注册的自动类名

**说明**:
- 允许tokenizer被AutoTokenizer自动发现和加载
- 用于自动模型加载机制

---

## 总结

`PreTrainedTokenizerBase` 类提供了完整的分词器功能框架，包括：

1. **完整的生命周期管理**: 从加载(`from_pretrained`)到保存(`save_pretrained`)
2. **灵活的编码选项**: 支持单文本、批量、序列对等多种编码场景
3. **现代聊天支持**: 完整的聊天模板和工具调用功能
4. **详细的配置选项**: 填充、截断、返回格式等全方位控制
5. **模型兼容性**: 支持各种张量框架和模型输入格式
6. **扩展性**: 抽象方法设计允许子类实现具体的分词逻辑

这个基类为所有Transformers分词器提供了统一而强大的接口，使得不同模型的分词器具有一致的API和行为。