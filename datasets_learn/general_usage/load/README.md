# 从 hub 中下载数据

数据集默认将所有数据加载到 train 集，或检查数据文件名称中是否存在提及或拆分名称（例如“train”、“test”和“validation”）。使用 data_files 参数将数据文件映射到 train 、 validation 和 test 等拆分

使用 data_files 或 data_dir 参数加载特定的文件子集，接受相对路径

# 本地和远程文件

数据集通常存储为 csv 、 json 、 txt 或 parquet 文件

最高效的格式是包含多个 JSON 对象；每行代表一行数据，嵌套字段，在这种情况下，需要指定 field 参数

Parquet 文件采用列式存储，这与 CSV 等基于行的文件不同。大型数据集可以存储在 Parquet 文件中，效率更高，查询返回速度也更快。

Arrow 文件以内存列式格式存储，与 CSV 等基于行的格式和 Parquet 等未压缩格式不同。Datasets 在后台使用的文件格式

WebDataset 格式基于 TAR 档案，适用于大型图像数据集。由于 WebDataset 的大小，通常以流式加载（使用 streaming=True ）

当数据集由多个文件（分片）组成时，多进程处理，每个进程分配一个分片子集加快处理

# 内存数据

直接从内存数据结构（如 Python 字典和 Pandas DataFrames）创建数据集

使用 from_dict() 加载 Python 字典、使用 from_list() 加载 Python 字典列表

使用 from_generator() 从 Python 生成器创建数据集，加载大于可用内存的数据

使用 from_pandas() 加载 Pandas DataFrames

# 离线

从 hub 下载的数据集会被缓存，环境变量 HF_HUB_OFFLINE 设置为 1 即可启用完全离线模式

# 切片分割

分片有两种方式：使用字符串或 ReadInstruction API。对于简单的情况，字符串更紧凑、更易读；而 ReadInstruction 更易于使用，因为它支持可变的分片参数

分割的特定行、选择分割百分比、从每个分割中选择百分比组合、创建交叉验证分割

# 指定特征

从本地文件创建数据集时， Apache Arrow 会自动推断特征，通过 calsslabel 自定义特征