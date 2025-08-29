# Datasets Arrow

Arrow 可以快速处理和移动大量数据。它是一种特定的数据格式，将数据存储在列式内存布局中。这提供了几个显著的优势：
- 允许零拷贝读取， 从而消除了几乎所有的序列化开销
- 与语言无关，支持不同的编程语言
- 面向列的，因此查询和处理数据切片或列的速度更快
- 允许无复制地切换到标准机器学习工具，例如 NumPy、Pandas、PyTorch 和 TensorFlow
- 支持多种（可能嵌套的）列类型

内存映射：Datasets 使用 Arrow 作为其本地缓存系统。它允许数据集由磁盘缓存支持，该缓存采用内存映射，以便快速查找。这种架构允许在设备内存相对较小的机器上使用大型数据集

例如，加载完整的英文维基百科数据集仅需要几 MB 的 RAM，这是可能的，因为 Arrow 数据实际上是从磁盘映射到内存的，而不是加载到内存中。内存映射允许访问磁盘上的数据，并利用虚拟内存功能进行快速查找。

# Cache

缓存是 Datasets 高效的原因之一。它存储之前下载并处理过的数据集，因此当再次需要使用它们时，它们会直接从缓存中重新加载。这避免了重新下载数据集或重新应用处理函数。

Datasets 重新计算所有内容的一个例子是禁用缓存时。发生这种情况时，每次都会生成缓存文件，并将其写入临时目录。Python 会话结束后，临时目录中的缓存文件将被删除。这些缓存文件会被分配一个随机哈希值，而不是指纹。

通过对传递给 map 的函数以及 map 参数（ batch_size 、 remove_columns 等）进行散列来更新数据集的指纹。

使用 finger.Hasher 检查任何 Python 对象的哈希值，哈希值的计算方法是使用 dill pickler 转储对象，然后对转储的字节进行哈希处理。pickler 会递归地转储函数中使用的所有变量，因此，对函数中使用的对象进行的任何更改都会导致哈希值发生变化。

某个函数在不同会话中的哈希值似乎不一致，则意味着至少有一个变量包含一个不确定的 Python 对象

# Dataset and IterableDataset

Features 定义了数据集的内部结构，用于指定底层序列化格式。更有趣的是， Features 包含从列名、类型到 ClassLabel 等所有内容的高级信息。 Features 为数据集的骨干。dict[column_name, column_type] 。它是一个包含列名和列类型对的字典。

检索标签时， ClassLabel.int2str() 和 ClassLabel.str2int() 会将整数值转换为标签名称

数组 (Array) 特性类型可用于创建各种大小的数组。可以使用 Array2D 创建二维数组，甚至可以使用 Array5D 创建五维数组。

图像数据集有一个 Image 类型的列，它从以字节形式存储的图像中加载 PIL.Image 对象：

# Build and load

数据集是一个包含以下内容的目录：
- 一些通用格式的数据文件（JSON、CSV、Parquet、文本等）
- 名为 README.md 的数据集卡，其中包含有关数据集的文档以及用于定义数据集标签和配置的 YAML 标头

如果数据集仅包含数据文件，则 load_dataset() 会自动推断如何根据数据文件的扩展名（json、csv、parquet、txt 等）加载它们。在底层，🤗 Datasets 会根据数据文件格式使用相应的 DatasetBuilder。🤗 Datasets 中，每种数据文件格式都有一个对应的构建器：比如用于 Parquet 的 datasets.packaged_modules.parquet.Parquet

🤗 Datasets 会从原始 URL 下载数据集文件，生成数据集并将其缓存到您硬盘上的 Arrow 表中。如果之前下载过该数据集，🤗 Datasets 会从缓存中重新加载

首次加载数据集时，🤗 Datasets 会获取原始数据文件，并将其构建为包含行和类型化列的表。有两个主要类负责构建数据集： BuilderConfig 和 DatasetBuilder 。

BuilderConfig 是 DatasetBuilder 的配置类， BuilderConfig 包含数据集的以下基本属性：

想向数据集添加其他属性（例如类标签），可以创建 BuilderConfig 基类的子类

DatasetBuilder 访问 BuilderConfig 内的所有属性来构建实际的数据集。
- DatasetBuilder._info() 负责定义数据集的属性
- DatasetBuilder._split_generator 下载或检索请求的数据文件， DownloadManager ，用于下载文件或从本地文件系统获取文件， SplitGenerator 会将它们整理成分片
- DatasetBuilder._generate_examples 读取并解析用于拆分的数据文件




