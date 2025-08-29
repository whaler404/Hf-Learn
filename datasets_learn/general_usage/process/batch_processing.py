from datasets import load_dataset
dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
batched_dataset = dataset.batch(batch_size=4)
print(batched_dataset[0])
# {'text': ['the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .',
#         'the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson\'s expanded vision of j . r . r . tolkien\'s middle-earth .',
#         'effective but too-tepid biopic',
#         'if you sometimes like to go to the movies to have fun , wasabi is a good place to start .'],
# 'label': [1, 1, 1, 1]}

