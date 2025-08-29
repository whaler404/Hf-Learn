数据集流式传输无需下载即可使用数据集。当迭代数据集时，数据会被流式传输，适合
- 不想等待下载非常大的数据集。
- 数据集大小超出了计算机上可用的磁盘空间量
- 快速探索数据集的几个样本

# 列索引

IterableDataset 支持列索引

IterableDataset.rename_column() ，与原始列关联的特征实际上会移动到新的列名下，而不是仅仅替换原始列。

IterableDataset.remove_columns() 函数删除多列，提供列名列表

# 从 Dataset 转化

已有 Dataset 对象，可以使用 to_iterable_dataset() 函数将其转换为 IterableDataset 。这实际上比在 load_dataset() 中设置 streaming=True 参数更快，因为数据是从本地文件流式传输的。实例化时支持分片。这在处理大型数据集，并且希望对数据集进行混排或使用 PyTorch DataLoader 实现快速并行加载时非常有用。

to_iterable_dataset() 函数在 IterableDataset 实例化时支持分片。这在处理大型数据集，并且希望对数据集进行混排或使用 PyTorch DataLoader 实现快速并行加载时非常有用。

IterableDataset.cast() 用于更改一列或多列的特征类型，新的 Features 作为参数

# 行操作

buffer_size 参数控制用于随机采样样本的缓冲区大小。假设数据集包含一百万个样本，并将 buffer_size 设置为一万个。IterableDataset.shuffle () 将从缓冲区的前一万个样本中随机选择样本。缓冲区中被选中的样本将被新的样本替换。

如果数据集被分片为多个文件， IterableDataset.shuffle() 也会打乱分片的顺序。

每个 epoch 之后重新整理数据集。这需要为每个 epoch 设置不同的种子。在 epoch 之间使用 IterableDataset.set_epoch() 来告知数据集当前处于哪个 epoch。

IterableDataset.take() 返回数据集中的前 n 示例，take 和 skip 会阻止后续对 shuffle 的调用，因为它们会锁定分片的顺序。应该在拆分数据集之前对其 shuffle 。

Datasets 支持分片功能，可以将超大数据集划分为预定义数量的数据块。在 shard() 函数中指定 num_shards 参数，确定数据集要划分的分片数量。使用 index 参数指定要返回的分片。

interleave_datasets() 可以将 IterableDataset 与其他数据集组合。组合后的数据集将返回来自每个原始数据集的交替样本。定义每个原始数据集的采样概率，控制每个数据集的采样和组合方式。

# map 映射

IterableDataset.map() 来处理 IterableDataset 。 当示例流式传输时， IterableDataset.map() 会即时应用处理

每个示例单独或批量地应用处理函数。该函数甚至可以创建新的行和列

设置 batched=True 可以对批量样本进行操作

batch 方法将 IterableDataset 转换为包含多个批次的可迭代对象。在训练循环中使用批次，或者使用需要批量输入的框架时，此功能尤其有用。batched_dataset 仍然是 IterableDataset

# 训练循环中的 Stream

IterableDataset 可以集成到训练循环中。首先，对数据集进行 shuffle ，

保存数据集检查点并恢复迭代：如果训练循环停止，从原来的位置重新开始训练。为此，保存模型、优化器以及数据加载器的检查点。

可迭代数据集不提供对特定示例索引的随机访问以从中恢复，但可以使用 IterableDataset.state_dict() 和 IterableDataset.load_state_dict() 从检查点恢复，类似于对模型和优化器所做的操作：

在底层，可迭代数据集跟踪当前正在读取的分片和当前分片中的示例索引，并将此信息存储在 state_dict 中。

要从检查点恢复，数据集会跳过所有之前已读取的分片，从当前分片重新启动。然后，它会读取该分片并跳过示例，直到到达检查点对应的确切示例。

因此，重启数据集非常快，因为它不会重新读取已经迭代过的分片。然而，恢复数据集通常不是即时的，因为它必须从当前分片的开头重新开始读取，并跳过示例直到到达检查点位置。

