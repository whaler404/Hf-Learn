from datasets import Features, Value, ClassLabel

# define your own labels with the Features class
class_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
emotion_features = Features({'text': Value('string'), 'label': ClassLabel(names=class_names)})

from datasets import load_dataset

file_dict = {"train": "train.csv", "test": "test.csv"}
# specify the features parameter in load_dataset() with the features
dataset = load_dataset('csv', data_files=file_dict, delimiter=';', column_names=['text', 'label'], features=emotion_features)

print(dataset['train'].features)