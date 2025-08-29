from datasets import load_dataset

# dataset is 45 terabytes, but you can use it instantly with streaming
dataset = load_dataset('HuggingFaceFW/fineweb', split='train', streaming=True)
print(next(iter(dataset)))
# {
#   "text": "How AP reported in all formats from tornado-stricken regionsMarch 8, ...",
#   "id": "<urn:uuid:d66bc6fe-8477-4adf-b430-f6a558ccc8ff>",
#   "dump": "CC-MAIN-2013-20",
#   "url": "http:// jwashington@ap.org/Content/Press-Release/2012/How-AP-reported-in-all-formats-from-tornado-stricken-regions",
#   "date": "2013-05-18T05:48:54Z",
#   "file_path": "s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz",
#   "language": "en",
#   "language_score": 0.9721424579620361,
#   "token_count": 717
# }

# rename column
from datasets import load_dataset
dataset = load_dataset('allenai/c4', 'en', streaming=True, split='train')
dataset = dataset.rename_column("text", "content")

# remove column
from datasets import load_dataset
dataset = load_dataset('allenai/c4', 'en', streaming=True, split='train')
dataset = dataset.remove_columns('timestamp')

# cast features
from datasets import load_dataset
dataset = load_dataset('nyu-mll/glue', 'mrpc', split='train', streaming=True)
print(dataset.features)
# {'sentence1': Value('string'),
# 'sentence2': Value('string'),
# 'label': ClassLabel(names=['not_equivalent', 'equivalent']),
# 'idx': Value('int32')}

from datasets import ClassLabel, Value
new_features = dataset.features.copy()
new_features["label"] = ClassLabel(names=['negative', 'positive'])
new_features["idx"] = Value('int64')
dataset = dataset.cast(new_features)
print(dataset.features)
# {'sentence1': Value('string'),
# 'sentence2': Value('string'),
# 'label': ClassLabel(names=['negative', 'positive']),
# 'idx': Value('int64')}

