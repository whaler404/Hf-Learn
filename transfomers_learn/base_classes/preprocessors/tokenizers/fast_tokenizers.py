from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

from tokenizers.pre_tokenizers import Whitespace

tokenizer.pre_tokenizer = Whitespace()

files = ['./datasets/text.txt']
tokenizer.train(files, trainer)

tokenizer.save("./datasets/tokenizer.json")

from transformers import PreTrainedTokenizerFast

# fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="./datasets/tokenizer.json")