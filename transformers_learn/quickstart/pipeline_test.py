from transformers import pipeline

# # 1. This script demonstrates how to use the Hugging Face Transformers library's pipeline feature

# classifier = pipeline("sentiment-analysis")
# classifier("We are very happy to show you the 🤗 Transformers library.")
# results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
# for result in results:
#     print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

# # 2. load the dataset and use the pipeline for automatic speech recognition
# import torch
# from transformers import pipeline

# speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
# from datasets import load_dataset, Audio

# dataset = load_dataset("PolyAI/minds14", name="en-US", split="train", trust_remote_code=True)
# # dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))

# result = speech_recognizer(dataset[:4]["audio"])
# print([d["text"] for d in result])

# 3. use another model and tokenizer for sentiment analysis
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
result = classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
print(result)