# Hugging Face Pipeline 支持的任务

Hugging Face Transformers 提供了丰富的 pipeline 接口，支持各种机器学习任务。本文档按数据类型对所有支持的任务进行分类。

## 目录
- [音频任务 (Audio)](#音频任务-audio)
- [文本任务 (Text)](#文本任务-text)
- [图像任务 (Image)](#图像任务-image)
- [多模态任务 (Multimodal)](#多模态任务-multimodal)
- [视频任务 (Video)](#视频任务-video)

## 音频任务 (Audio)

| Pipeline 名称 | 实现类 | PT 模型类型 | 默认模型 |
|---------------|--------|------------|----------|
| `audio-classification` | `AudioClassificationPipeline` | `AutoModelForAudioClassification` | `superb/wav2vec2-base-superb-ks` |
| `text-to-audio` | `TextToAudioPipeline` | `AutoModelForTextToWaveform`, `AutoModelForTextToSpectrogram` | `suno/bark-small` |

### 任务别名
- `text-to-speech` → `text-to-audio`

## 文本任务 (Text)

| Pipeline 名称 | 实现类 | PT 模型类型 | TF 模型类型 | 默认模型 (PT) | 默认模型 (TF) |
|---------------|--------|------------|------------|---------------|---------------|
| `text-classification` | `TextClassificationPipeline` | `AutoModelForSequenceClassification` | `TFAutoModelForSequenceClassification` | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` |
| `token-classification` | `TokenClassificationPipeline` | `AutoModelForTokenClassification` | `TFAutoModelForTokenClassification` | `dbmdz/bert-large-cased-finetuned-conll03-english` | `dbmdz/bert-large-cased-finetuned-conll03-english` |
| `question-answering` | `QuestionAnsweringPipeline` | `AutoModelForQuestionAnswering` | `TFAutoModelForQuestionAnswering` | `distilbert/distilbert-base-cased-distilled-squad` | `distilbert/distilbert-base-cased-distilled-squad` |
| `table-question-answering` | `TableQuestionAnsweringPipeline` | `AutoModelForTableQuestionAnswering` | `TFAutoModelForTableQuestionAnswering` | `google/tapas-base-finetuned-wtq` | `google/tapas-base-finetuned-wtq` |
| `fill-mask` | `FillMaskPipeline` | `AutoModelForMaskedLM` | `TFAutoModelForMaskedLM` | `distilbert/distilroberta-base` | `distilbert/distilroberta-base` |
| `summarization` | `SummarizationPipeline` | `AutoModelForSeq2SeqLM` | `TFAutoModelForSeq2SeqLM` | `sshleifer/distilbart-cnn-12-6` | `google-t5/t5-small` |
| `translation` | `TranslationPipeline` | `AutoModelForSeq2SeqLM` | `TFAutoModelForSeq2SeqLM` | `google-t5/t5-base` | `google-t5/t5-base` |
| `text2text-generation` | `Text2TextGenerationPipeline` | `AutoModelForSeq2SeqLM` | `TFAutoModelForSeq2SeqLM` | `google-t5/t5-base` | `google-t5/t5-base` |
| `text-generation` | `TextGenerationPipeline` | `AutoModelForCausalLM` | `TFAutoModelForCausalLM` | `openai-community/gpt2` | `openai-community/gpt2` |
| `zero-shot-classification` | `ZeroShotClassificationPipeline` | `AutoModelForSequenceClassification` | `TFAutoModelForSequenceClassification` | `facebook/bart-large-mnli` | `FacebookAI/roberta-large-mnli` |

### 翻译任务支持的语言对
- `en` → `fr` (英语到法语)
- `en` → `de` (英语到德语)
- `en` → `ro` (英语到罗马尼亚语)

### 任务别名
- `sentiment-analysis` → `text-classification`
- `ner` → `token-classification`

## 图像任务 (Image)

| Pipeline 名称 | 实现类 | PT 模型类型 | TF 模型类型 | 默认模型 (PT) | 默认模型 (TF) |
|---------------|--------|------------|------------|---------------|---------------|
| `image-classification` | `ImageClassificationPipeline` | `AutoModelForImageClassification` | `TFAutoModelForImageClassification` | `google/vit-base-patch16-224` | `google/vit-base-patch16-224` |
| `image-feature-extraction` | `ImageFeatureExtractionPipeline` | `AutoModel` | `TFAutoModel` | `google/vit-base-patch16-224` | `google/vit-base-patch16-224` |
| `depth-estimation` | `DepthEstimationPipeline` | `AutoModelForDepthEstimation` | 无 | `Intel/dpt-large` | 无 |
| `image-to-image` | `ImageToImagePipeline` | `AutoModelForImageToImage` | 无 | `caidas/swin2SR-classical-sr-x2-64` | 无 |
| `keypoint-matching` | `KeypointMatchingPipeline` | `AutoModelForKeypointMatching` | 无 | `magic-leap-community/superglue_outdoor` | 无 |

## 多模态任务 (Multimodal)

| Pipeline 名称 | 实现类 | PT 模型类型 | TF 模型类型 | 默认模型 |
|---------------|--------|------------|------------|----------|
| `automatic-speech-recognition` | `AutomaticSpeechRecognitionPipeline` | `AutoModelForCTC`, `AutoModelForSpeechSeq2Seq` | 无 | `facebook/wav2vec2-base-960h` |
| `feature-extraction` | `FeatureExtractionPipeline` | `AutoModel` | `TFAutoModel` | `distilbert/distilbert-base-cased` |
| `visual-question-answering` | `VisualQuestionAnsweringPipeline` | `AutoModelForVisualQuestionAnswering` | 无 | `dandelin/vilt-b32-finetuned-vqa` |
| `document-question-answering` | `DocumentQuestionAnsweringPipeline` | `AutoModelForDocumentQuestionAnswering` | 无 | `impira/layoutlm-document-qa` |
| `zero-shot-image-classification` | `ZeroShotImageClassificationPipeline` | `AutoModelForZeroShotImageClassification` | `TFAutoModelForZeroShotImageClassification` | `openai/clip-vit-base-patch32` |
| `zero-shot-audio-classification` | `ZeroShotAudioClassificationPipeline` | `AutoModel` | 无 | `laion/clap-htsat-fused` |
| `image-segmentation` | `ImageSegmentationPipeline` | `AutoModelForImageSegmentation`, `AutoModelForSemanticSegmentation` | 无 | `facebook/detr-resnet-50-panoptic` |
| `image-to-text` | `ImageToTextPipeline` | `AutoModelForVision2Seq` | `TFAutoModelForVision2Seq` | `ydshieh/vit-gpt2-coco-en` |
| `image-text-to-text` | `ImageTextToTextPipeline` | `AutoModelForImageTextToText` | 无 | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` |
| `object-detection` | `ObjectDetectionPipeline` | `AutoModelForObjectDetection` | 无 | `facebook/detr-resnet-50` |
| `zero-shot-object-detection` | `ZeroShotObjectDetectionPipeline` | `AutoModelForZeroShotObjectDetection` | 无 | `google/owlvit-base-patch32` |
| `mask-generation` | `MaskGenerationPipeline` | `AutoModelForMaskGeneration` | 无 | `facebook/sam-vit-huge` |

### 任务别名
- `vqa` → `visual-question-answering`

## 视频任务 (Video)

| Pipeline 名称 | 实现类 | PT 模型类型 | 默认模型 |
|---------------|--------|------------|----------|
| `video-classification` | `VideoClassificationPipeline` | `AutoModelForVideoClassification` | `MCG-NJU/videomae-base-finetuned-kinetics` |

---

## 使用示例

### 基本用法

```python
from transformers import pipeline

# 文本分类
classifier = pipeline("text-classification")
result = classifier("This movie is fantastic!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# 图像分类
image_classifier = pipeline("image-classification")
result = image_classifier("path/to/image.jpg")
print(result)  # [{'label': 'tabby', 'score': 0.9123}, ...]

# 问答
qa_pipeline = pipeline("question-answering")
result = qa_pipeline(
    question="Where is the Eiffel Tower?",
    context="The Eiffel Tower is in Paris, France."
)
print(result)  # {'score': 0.999, 'start': 21, 'end': 31, 'answer': 'Paris, France'}
```

### 指定模型

```python
# 使用特定模型
sentiment_pipeline = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# 零样本分类
zero_shot_classifier = pipeline("zero-shot-classification")
result = zero_shot_classifier(
    "I love playing video games",
    candidate_labels=["entertainment", "sports", "technology"],
    hypothesis_template="This text is about {}."
)
print(result)
```

### 多模态任务

```python
# 图像到文本
image_to_text = pipeline("image-to-text")
result = image_to_text("path/to/image.jpg")
print(result)  # [{'generated_text': 'A cat sitting on a sofa'}]

# 视觉问答
vqa_pipeline = pipeline("visual-question-answering")
result = vqa_pipeline(
    image="path/to/image.jpg",
    question="What color is the cat?"
)
print(result)  # [{'score': 0.92, 'answer': 'black'}]
```

---

## 模型框架支持

- **PT (PyTorch)**: 原生支持，推荐使用
- **TF (TensorFlow)**: 大多数任务支持，部分任务无TensorFlow实现

---

## 任务说明

### 音频任务
- **音频分类**: 对音频片段进行分类（如音乐类型、语音命令识别）
- **文本转音频**: 将文本转换为语音（TTS）

### 文本任务
- **文本分类**: 对整个文本进行分类（情感分析、主题分类）
- **词元分类**: 对文本中的每个词元进行分类（命名实体识别、词性标注）
- **问答**: 基于给定上下文回答问题
- **表格问答**: 基于表格数据回答问题
- **填空**: 预测文本中被掩码的词
- **摘要**: 生成文本摘要
- **翻译**: 在不同语言间翻译文本
- **文本生成**: 生成连贯的文本
- **零样本分类**: 在没有训练数据的情况下对文本分类
- **文本到文本**: 通用的序列到序列任务

### 图像任务
- **图像分类**: 对整个图像进行分类
- **图像特征提取**: 提取图像的表示特征
- **深度估计**: 估计图像中物体的深度
- **图像到图像**: 图像转换任务（超分辨率、去噪等）
- **关键点匹配**: 匹配图像间的关键点

### 多模态任务
- **自动语音识别**: 将语音转换为文本
- **特征提取**: 通用特征提取（支持多种模态）
- **视觉问答**: 基于图像回答问题
- **文档问答**: 基于文档图像回答问题
- **零样本图像分类**: 对图像进行分类而无需训练数据
- **零样本音频分类**: 对音频进行分类而无需训练数据
- **图像分割**: 对图像进行像素级分割
- **图像到文本**: 为图像生成描述文本
- **图像文本到文本**: 基于图像和文本生成文本
- **目标检测**: 在图像中检测和定位物体
- **零样本目标检测**: 检测物体而无需特定的训练数据
- **掩码生成**: 生成图像的分割掩码

### 视频任务
- **视频分类**: 对视频片段进行分类