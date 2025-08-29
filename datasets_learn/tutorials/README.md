# 从 Hub 加载数据集

数据集的信息存储在 `DatasetInfo` 中，包含数据集描述、特征和数据集大小等信息。使用 `load_dataset_builder()` 函数加载数据集构建器并检查数据集的属性，而无需下载它

split 是数据集的特定子集，例如 train 和 test 。加载数据 split 将返回一个 `Dataset` 对象

数据集如果包含多个子数据集，称为 `configuration`

数据集对象有两种类型：常规数据集 `Dataset` 和 `IterableDataset` 。 数据集提供对行的快速随机访问和内存映射，因此即使加载大型数据集也仅占用相对较小的设备内存。但对于磁盘或内存都无法容纳的超大数据集， IterableDataset 无需等待数据集完全下载即可访问和使用

数据集包含多列数据，每列可以是不同类型的数据。 索引 （或称轴标签）用于访问数据集中的示例

IterableDataset 的行为与常规 Dataset 不同。无法随机访问 IterableDataset 中的样本，应该迭代其元素

可以使用 IterableDataset.take() 返回包含特定数量示例的数据集子集

# 创建数据集

基于文件的构建器：数据集支持许多常见格式，例如 csv ， json/jsonl ， parquet ， txt

基于文件夹的构建器，用于快速创建图像或音频数据集，`folder/split/label/image`

`from_` 方法用于从本地文件创建数据集

# 共享数据集

元数据 UI ，其中包含多个字段可供选择，例如许可证、语言和任务类别