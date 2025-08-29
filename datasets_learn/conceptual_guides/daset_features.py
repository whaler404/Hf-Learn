from datasets import load_dataset_builder
dataset_builder = load_dataset_builder('rajpurkar/squad', split='train')
print(dataset_builder.info.features)
# {'id': Value('string'),
#  'title': Value('string'),
#  'context': Value('string'),
#  'question': Value('string'),
#  'answers': {'text': List(Value('string')),
#   'answer_start': List(Value('int32'))}}

# image feature
from datasets import load_dataset, Image

dataset = load_dataset("AI-Lab-Makerere/beans", split="train")
dataset[0]["image"]
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x500 at 0x125506CF8>
