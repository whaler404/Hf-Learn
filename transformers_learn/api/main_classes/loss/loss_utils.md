# Transformers 损失函数详解

本文档详细介绍了 Hugging Face Transformers 库中各种损失函数的实现，包括代码分析、张量形状变化和数学公式推导。

## 目录
- [fixed_cross_entropy](#fixed_cross_entropy)
- [ForCausalLMLoss](#forcausallmloss)
- [ForMaskedLMLoss](#formaskedlmloss)
- [ForSequenceClassificationLoss](#forsequenceclassificationloss)
- [ForQuestionAnsweringLoss](#forquestionansweringloss)
- [ForTokenClassification](#fortokenclassification)
- [损失函数映射表](#损失函数映射表)

---

## fixed_cross_entropy

### 功能说明
固定交叉熵损失函数，支持批量处理和忽略特定索引。

### 代码分析

```python
def fixed_cross_entropy(
    source: torch.Tensor,                # [N, vocab_size] - 预测的logits，N为批次中的有效token数
    target: torch.Tensor,                # [N] - 真实标签，N为批次中的有效token数
    num_items_in_batch: Optional[torch.Tensor] = None,  # [1] - 批次中的有效项目数量
    ignore_index: int = -100,            # 要忽略的标签索引
    **kwargs,
) -> torch.Tensor:                      # 返回: [1] - 标量损失值
    # 根据是否提供批次大小选择缩减方式
    reduction = "sum" if num_items_in_batch is not None else "mean"  # str

    # 计算交叉熵损失
    # source: [N, vocab_size] - 预测概率分布
    # target: [N] - 真实标签
    loss = nn.functional.cross_entropy(
        source,                          # [N, vocab_size] - 输入logits
        target,                          # [N] - 目标标签
        ignore_index=ignore_index,       # 忽略特定索引
        reduction=reduction              # 缩减方式
    )                                    # loss: [1] - 缩减后的损失值

    # 如果是求和模式，需要除以有效项目数量
    if reduction == "sum":               # 如果使用求和缩减
        # 确保num_items_in_batch是张量并移到正确设备
        if torch.is_tensor(num_items_in_batch):  # [1] - 检查是否为张量
            num_items_in_batch = num_items_in_batch.to(loss.device)  # [1] 移到与loss相同设备

        # 计算平均损失
        loss = loss / num_items_in_batch  # [1] / [1] -> [1] - 标量

    return loss                          # [1] - 最终损失值
```

### 数学公式

对于单个样本的交叉熵损失：

$$\text{CE}(y, \hat{y}) = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

其中：
- $y$ 是真实的one-hot编码标签向量
- $\hat{y}$ 是预测的softmax概率分布
- $C$ 是类别数

对于单个样本（使用标签索引表示）：

$$\text{CE}(y, \hat{y}) = -\log(\hat{y}_y)$$

对于批次损失（求平均）：

$$\text{Loss} = \frac{1}{N}\sum_{i=1}^{N} \text{CE}(y_i, \hat{y}_i)$$

其中 $N$ 是批次中有效样本数量。

---

## ForCausalLMLoss

### 功能说明
因果语言模型损失，用于自回归语言模型（如GPT）的训练。

### 代码分析

```python
def ForCausalLMLoss(
    logits,                               # [batch_size, seq_len, vocab_size] - 模型输出的logits
    labels,                               # [batch_size, seq_len] - 真实标签序列
    vocab_size: int,                      # 词汇表大小
    num_items_in_batch: Optional[torch.Tensor] = None,  # [1] - 批次中有效token数量
    ignore_index: int = -100,             # 要忽略的标签索引
    shift_labels: Optional[torch.Tensor] = None,  # [batch_size, seq_len-1] - 移位后的标签
    **kwargs,
) -> torch.Tensor:                       # 返回: [1] - 标量损失值

    # 转换为float32避免精度问题
    logits = logits.float()               # [batch_size, seq_len, vocab_size] - 转换数据类型

    if shift_labels is None:              # 如果没有提供移位标签
        # 在标签序列右侧填充一个ignore_index token
        labels = nn.functional.pad(
            labels,                       # [batch_size, seq_len] - 原始标签
            (0, 1),                      # 左侧填充0个，右侧填充1个
            value=ignore_index            # 填充值为ignore_index
        )                                 # labels: [batch_size, seq_len+1]

        # 移位：下一个token作为当前token的预测目标
        shift_labels = labels[..., 1:].contiguous()  # [batch_size, seq_len] - 取除了第一个token的所有标签

    # 将logits展平为2D张量，用于交叉熵计算
    logits = logits.view(-1, vocab_size) # [batch_size*seq_len, vocab_size] - 展平logits

    # 将移位标签展平为1D张量
    shift_labels = shift_labels.view(-1) # [batch_size*seq_len] - 展平标签

    # 确保标签在正确的设备上（支持模型并行）
    shift_labels = shift_labels.to(logits.device)  # [batch_size*seq_len] - 移到logits设备

    # 使用固定交叉熵计算损失
    loss = fixed_cross_entropy(
        logits,                           # [batch_size*seq_len, vocab_size] - 展平的logits
        shift_labels,                     # [batch_size*seq_len] - 展平的移位标签
        num_items_in_batch,               # [1] - 有效token数量
        ignore_index,                     # 忽略索引
        **kwargs
    )                                     # [1] - 最终损失值

    return loss                           # [1] - 标量损失值
```

### 数学公式

因果语言模型的目标是预测下一个token：

$$\text{Loss} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{S} \log P(y_{i,j+1} | y_{i,1:j})$$

其中：
- $N$ 是批次大小
- $S$ 是序列长度
- $y_{i,j}$ 是第 $i$ 个样本的第 $j$ 个token
- $P(y_{i,j+1} | y_{i,1:j})$ 是给定前面token预测下一个token的概率

---

## ForMaskedLMLoss

### 功能说明
掩码语言模型损失，用于BERT等掩码语言模型的训练。

### 代码分析

```python
def ForMaskedLMLoss(
    logits: torch.Tensor,                 # [batch_size, seq_len, vocab_size] - 模型输出的logits
    labels: torch.Tensor,                 # [batch_size, seq_len] - 真实标签，非掩码位置为ignore_index
    vocab_size: int,                      # 词汇表大小
    num_items_in_batch: Optional[torch.Tensor] = None,  # [1] - 批次中有效token数量
    ignore_index: int = -100,             # 要忽略的标签索引
    **kwargs,
) -> torch.Tensor:                       # 返回: [1] - 标量损失值

    # 转换为float32避免精度问题
    logits = logits.float()               # [batch_size, seq_len, vocab_size] - 转换数据类型

    # 将logits展平为2D张量
    logits = logits.view(-1, vocab_size) # [batch_size*seq_len, vocab_size] - 展平logits

    # 将标签展平为1D张量
    labels = labels.view(-1)              # [batch_size*seq_len] - 展平标签

    # 确保标签在正确的设备上（支持模型并行）
    labels = labels.to(logits.device)     # [batch_size*seq_len] - 移到logits设备

    # 使用固定交叉熵计算损失
    loss = fixed_cross_entropy(
        logits,                           # [batch_size*seq_len, vocab_size] - 展平的logits
        labels,                           # [batch_size*seq_len] - 展平的标签
        num_items_in_batch,               # [1] - 有效token数量
        ignore_index,                     # 忽略索引
        **kwargs
    )                                     # [1] - 最终损失值

    return loss                           # [1] - 标量损失值
```

### 数学公式

掩码语言模型只对被掩码的token计算损失：

$$\text{Loss} = -\frac{1}{M}\sum_{i=1}^{N}\sum_{j \in M_i} \log P(y_{i,j} | x_{i,\neg j})$$

其中：
- $N$ 是批次大小
- $M_i$ 是第 $i$ 个样本中被掩码的位置集合
- $M = \sum_{i=1}^{N}|M_i|$ 是被掩码token的总数
- $x_{i,\neg j}$ 表示除了位置 $j$ 之外的所有token
- $P(y_{i,j} | x_{i,\neg j})$ 是给定上下文预测被掩码token的概率

---

## ForSequenceClassificationLoss

### 功能说明
序列分类损失，支持回归、单标签分类和多标签分类三种任务。

### 代码分析

```python
def ForSequenceClassificationLoss(
    labels: torch.Tensor,                 # [batch_size] - 真实标签
    pooled_logits: torch.Tensor,          # [batch_size, num_labels] - 池化后的logits
    config,                               # 模型配置对象
    **kwargs
) -> torch.Tensor:                       # 返回: [1] - 标量损失值

    num_labels = config.num_labels        # int - 类别数量

    # 自动确定问题类型
    if config.problem_type is None:       # 如果没有指定问题类型
        if num_labels == 1:               # 如果只有一个类别
            config.problem_type = "regression"      # 设置为回归问题
        elif num_labels > 1 and (labels.dtype in (torch.long, torch.int)):  # 多个类别且标签为整数类型
            config.problem_type = "single_label_classification"  # 单标签分类
        else:                             # 其他情况
            config.problem_type = "multi_label_classification"     # 多标签分类

    # 确保标签在正确的设备上
    labels = labels.to(pooled_logits.device)  # [batch_size] - 移到logits设备

    if config.problem_type == "regression":   # 回归问题
        loss_fct = MSELoss()               # 创建均方误差损失函数

        if num_labels == 1:               # 单变量回归
            # 移除多余维度并计算MSE损失
            return loss_fct(
                pooled_logits.squeeze(),  # [batch_size] -> [batch_size] - 移除最后一维
                labels.squeeze()          # [batch_size] -> [batch_size] - 移除最后一维
            )                             # [1] - MSE损失值
        else:                             # 多变量回归
            # 直接计算MSE损失
            return loss_fct(
                pooled_logits,            # [batch_size, num_labels]
                labels                    # [batch_size, num_labels]
            )                             # [1] - MSE损失值

    if config.problem_type == "single_label_classification":  # 单标签分类
        # 使用固定交叉熵计算损失
        return fixed_cross_entropy(
            pooled_logits.view(-1, num_labels),  # [batch_size, num_labels] - 展平logits
            labels.view(-1),                     # [batch_size] - 展平标签
            **kwargs
        )                                       # [1] - 交叉熵损失值

    if config.problem_type == "multi_label_classification":  # 多标签分类
        # 使用带逻辑斯的二元交叉熵损失
        loss_fct = BCEWithLogitsLoss()      # 创建BCE损失函数
        return loss_fct(
            pooled_logits,                  # [batch_size, num_labels] - 原始logits
            labels                          # [batch_size, num_labels] - 多标签目标
        )                                   # [1] - BCE损失值

    # 如果问题类型无效，抛出异常
    raise RuntimeError(f"Invalid problem type: {config.problem_type}")
```

### 数学公式

**1. 回归问题（均方误差）：**

$$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

其中：
- $N$ 是批次大小
- $y_i$ 是真实值
- $\hat{y}_i$ 是预测值

**2. 单标签分类（交叉熵）：**

$$\text{CE} = -\frac{1}{N}\sum_{i=1}^{N}\log P(y_i | x_i)$$

其中：
- $P(y_i | x_i)$ 是给定输入 $x_i$ 预测类别 $y_i$ 的概率

**3. 多标签分类（二元交叉熵）：**

$$\text{BCE} = -\frac{1}{N \times C}\sum_{i=1}^{N}\sum_{j=1}^{C} [y_{i,j}\log \sigma(\hat{y}_{i,j}) + (1-y_{i,j})\log(1-\sigma(\hat{y}_{i,j}))]$$

其中：
- $N$ 是批次大小
- $C$ 是类别数量
- $\sigma$ 是sigmoid函数
- $y_{i,j} \in \{0,1\}$ 是第 $i$ 个样本第 $j$ 个类别的真实标签
- $\hat{y}_{i,j}$ 是第 $i$ 个样本第 $j$ 个类别的logits

---

## ForQuestionAnsweringLoss

### 功能说明
问答任务损失，同时预测答案的起始位置和结束位置。

### 代码分析

```python
def ForQuestionAnsweringLoss(
    start_logits,                         # [batch_size, seq_len] - 起始位置logits
    end_logits,                           # [batch_size, seq_len] - 结束位置logits
    start_positions,                      # [batch_size] - 真实起始位置
    end_positions,                        # [batch_size] - 真实结束位置
    **kwargs
) -> torch.Tensor:                       # 返回: [1] - 标量损失值

    total_loss = None                     # 初始化总损失

    # 检查是否提供了起始和结束位置
    if start_positions is not None and end_positions is not None:

        # 处理多GPU情况：如果张量有多余维度，移除最后一维
        if len(start_positions.size()) > 1:      # [batch_size, 1] -> 检查维度
            start_positions = start_positions.squeeze(-1).to(start_logits.device)  # [batch_size] - 移除并移设备

        if len(end_positions.size()) > 1:        # [batch_size, 1] -> 检查维度
            end_positions = end_positions.squeeze(-1).to(end_logits.device)      # [batch_size] - 移除并移设备

        # 处理超出范围的位置：将位置限制在有效范围内
        ignored_index = start_logits.size(1)      # int - 序列长度，用作忽略索引
        start_positions = start_positions.clamp(0, ignored_index)  # [batch_size] - 限制范围[0, seq_len]
        end_positions = end_positions.clamp(0, ignored_index)      # [batch_size] - 限制范围[0, seq_len]

        # 计算起始位置损失
        start_loss = fixed_cross_entropy(
            start_logits,                     # [batch_size, seq_len] - 起始logits
            start_positions,                  # [batch_size] - 起始位置标签
            ignore_index=ignored_index,       # 忽略超出范围的索引
            **kwargs
        )                                     # [1] - 起始位置损失

        # 计算结束位置损失
        end_loss = fixed_cross_entropy(
            end_logits,                       # [batch_size, seq_len] - 结束logits
            end_positions,                    # [batch_size] - 结束位置标签
            ignore_index=ignored_index,       # 忽略超出范围的索引
            **kwargs
        )                                     # [1] - 结束位置损失

        # 计算总损失（起始和结束位置损失的平均）
        total_loss = (start_loss + end_loss) / 2  # ([1] + [1]) / 2 -> [1]

    return total_loss                        # [1] - 最终损失值（可能为None）
```

### 数学公式

问答任务损失是起始位置损失和结束位置损失的平均值：

$$\text{Loss} = \frac{1}{2}(\text{CE}_{\text{start}} + \text{CE}_{\text{end}})$$

其中：

起始位置交叉熵损失：
$$\text{CE}_{\text{start}} = -\frac{1}{N}\sum_{i=1}^{N}\log P(s_i^* | x_i)$$

结束位置交叉熵损失：
$$\text{CE}_{\text{end}} = -\frac{1}{N}\sum_{i=1}^{N}\log P(e_i^* | x_i)$$

这里：
- $N$ 是批次大小
- $s_i^*$ 是第 $i$ 个样本的真实起始位置
- $e_i^*$ 是第 $i$ 个样本的真实结束位置
- $P(s_i^* | x_i)$ 是给定上下文预测起始位置的概率
- $P(e_i^* | x_i)$ 是给定上下文预测结束位置的概率

---

## ForTokenClassification

### 功能说明
序列标注（token分类）损失，用于命名实体识别、词性标注等任务。

### 代码分析

```python
def ForTokenClassification(
    logits: torch.Tensor,                  # [batch_size, seq_len, num_labels] - 每个token的logits
    labels,                                # [batch_size, seq_len] - 每个token的标签
    config,                                # 模型配置对象
    **kwargs
) -> torch.Tensor:                        # 返回: [1] - 标量损失值

    # 将logits展平为2D张量
    logits = logits.view(-1, config.num_labels)  # [batch_size*seq_len, num_labels] - 展平logits

    # 将标签展平为1D张量并移到正确设备
    labels = labels.view(-1).to(logits.device)  # [batch_size*seq_len] - 展平标签并移设备

    # 转换为float32避免精度问题
    logits = logits.float()               # [batch_size*seq_len, num_labels] - 转换数据类型

    # 使用固定交叉熵计算损失
    return fixed_cross_entropy(
        logits,                           # [batch_size*seq_len, num_labels] - 展平的logits
        labels,                           # [batch_size*seq_len] - 展平的标签
        **kwargs
    )                                     # [1] - 最终损失值
```

### 数学公式

序列标注损失是对每个token的分类损失求平均：

$$\text{Loss} = -\frac{1}{N \times S}\sum_{i=1}^{N}\sum_{j=1}^{S}\log P(y_{i,j} | x_{i,j})$$

其中：
- $N$ 是批次大小
- $S$ 是序列长度
- $y_{i,j}$ 是第 $i$ 个样本第 $j$ 个token的真实标签
- $P(y_{i,j} | x_{i,j})$ 是给定第 $j$ 个token的上下文预测其标签的概率

---

## 损失函数映射表

下表显示了不同的模型架构与对应损失函数的映射关系：

| 模型类型 | 损失函数 | 应用场景 |
|---------|---------|----------|
| `ForCausalLM` | `ForCausalLMLoss` | 自回归语言模型（GPT等） |
| `ForMaskedLM` | `ForMaskedLMLoss` | 掩码语言模型（BERT等） |
| `ForQuestionAnswering` | `ForQuestionAnsweringLoss` | 问答任务（BERT-QA等） |
| `ForSequenceClassification` | `ForSequenceClassificationLoss` | 序列分类任务 |
| `ForImageClassification` | `ForSequenceClassificationLoss` | 图像分类任务 |
| `ForVideoClassification` | `ForSequenceClassificationLoss` | 视频分类任务 |
| `ForAudioClassification` | `ForSequenceClassificationLoss` | 音频分类任务 |
| `ForTokenClassification` | `ForTokenClassification` | 序列标注任务（NER、POS等） |
| `ForSegmentation` | `ForSegmentationLoss` | 语义分割任务 |
| `ForObjectDetection` | `ForObjectDetectionLoss` | 目标检测任务 |
| `ForConditionalGeneration` | `ForCausalLMLoss` | 条件生成任务 |
| `DeformableDetrForObjectDetection` | `DeformableDetrForObjectDetectionLoss` | 可变形DETR目标检测 |
| `ConditionalDetrForObjectDetection` | `DeformableDetrForObjectDetectionLoss` | 条件DETR目标检测 |
| `DabDetrForObjectDetection` | `DeformableDetrForObjectDetectionLoss` | DabDETR目标检测 |
| `GroundingDinoForObjectDetection` | `GroundingDinoForObjectDetectionLoss` | Grounding DINO目标检测 |
| `MMGroundingDinoForObjectDetection` | `GroundingDinoForObjectDetectionLoss` | 多模态Grounding DINO |
| `ConditionalDetrForSegmentation` | `DeformableDetrForSegmentationLoss` | 条件DETR分割 |
| `RTDetrForObjectDetection` | `RTDetrForObjectDetectionLoss` | RT-DETR目标检测 |
| `RTDetrV2ForObjectDetection` | `RTDetrForObjectDetectionLoss` | RT-DETR v2目标检测 |
| `DFineForObjectDetection` | `DFineForObjectDetectionLoss` | DFine目标检测 |
| `CsmForConditionalGeneration` | `ForCausalLMLoss` | CSM条件生成 |

---

## 通用注意事项

1. **精度处理**: 所有损失函数都先将logits转换为float32以避免精度问题
2. **设备管理**: 自动处理张量在不同设备间的移动，支持模型并行
3. **忽略索引**: 支持通过`ignore_index`参数忽略特定的标签（通常是-100）
4. **批量处理**: 支持动态批量大小和有效项目数量的计算
5. **内存效率**: 使用展平操作优化内存使用效率

---

## 使用示例

```python
import torch
from transformers.loss.loss_utils import ForCausalLMLoss, ForSequenceClassificationLoss

# 因果语言模型损失示例
batch_size, seq_len, vocab_size = 2, 10, 1000
logits = torch.randn(batch_size, seq_len, vocab_size)
labels = torch.randint(0, vocab_size, (batch_size, seq_len))

causal_loss = ForCausalLMLoss(logits, labels, vocab_size)
print(f"Causal LM Loss: {causal_loss.item()}")

# 序列分类损失示例
batch_size, num_labels = 4, 3
pooled_logits = torch.randn(batch_size, num_labels)
classification_labels = torch.randint(0, num_labels, (batch_size,))

# 模拟配置对象
class Config:
    def __init__(self):
        self.num_labels = num_labels
        self.problem_type = None

config = Config()
classification_loss = ForSequenceClassificationLoss(classification_labels, pooled_logits, config)
print(f"Classification Loss: {classification_loss.item()}")
```