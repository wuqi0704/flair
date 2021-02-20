# %%
# import flair
from flair.datasets import DataLoader
# from typing import Union, List

# LanguageList = [
#     'HEBREW','ARABIC','PORTUGUESE','ITALIAN','FRENCH','SPANISH','GERMAN','ENGLISH',
#     'RUSSIAN','FINNISH','VIETNAMESE','KOREAN','CHINESE','JAPANESE'
# ]
LanguageList = [
    'ENGLISH'
]
import pickle

data_train, data_test, data_dev = [], [], []
for language in LanguageList:
    with open('resources/%s_Train.pickle' % language, 'rb') as f1:
        train = pickle.load(f1)
    with open('resources/%s_Test.pickle' % language, 'rb') as f2:
        test = pickle.load(f2)
    with open('resources/%s_Dev.pickle' % language, 'rb') as f3:
        dev = pickle.load(f3)

    data_train += train
    data_test += test
    data_dev += dev

letter_to_ix = {}
letter_to_ix[''] = 0  # need this for padding
for sent, tags in data_train + data_test + data_dev:
    for letter in sent:
        if letter not in letter_to_ix:
            letter_to_ix[letter] = len(letter_to_ix)
print('functions.py : Nr. of distinguish character: ', len(letter_to_ix.keys()))

tag_to_ix = {'B': 0, 'I': 1, 'E': 2, 'S': 3, 'X': 4}
ix_to_tag = {y: x for x, y in tag_to_ix.items()}


# %%
from flair.data import LabeledString

# import torch
# LabeledString is a DataPoint - init and set the label
sentence = LabeledString('Any major dischord and we all suffer.')
sentence.set_label('tokenization', 'BIEXBIIIEXBIIIIIIEXBIEXBEXBIEXBIIIIES')

sentence_2 = LabeledString('All upper airplane and or any suffer?')
sentence_2.set_label('tokenization', 'BIEXBIIIEXBIIIIIIEXBIEXBEXBIEXBIIIIES')

# Print the DataPoint
print(sentence)

# Print the string
print(sentence.string)

# print the label
print(sentence.get_labels('tokenization'))


#%%
# from flair.models.tokenizer_model import *
from flair.models.tokenizer_model import FlairTokenizer

# character_size = 10
embedding_dim = 4096
hidden_dim = 256
num_layers = 1
batch_size = 1
use_CSE = False
# init the tokenizer like you would your LSTMTagger
tokenizer: FlairTokenizer = FlairTokenizer(letter_to_ix, embedding_dim, hidden_dim, num_layers, batch_size,
                                           use_CSE)

# FIXME: do a forward pass and compute the loss for two data points
loss = tokenizer.forward_loss([sentence, sentence_2])

# tag_scores, loss = tokenizer.forward_loss(sentence_2)
# loss should be a single value tensor 
print(loss)

#%%
sentences = [sentence, sentence_2]
tag_scores = tokenizer.forward_loss(sentences,foreval=True)
tag_scores.shape

tokenizer.evaluate(sentences)

#%%
tag_scores.shape

# %% save the model
# filename = 'test.tar'
# checkpoint = tokenizer._get_state_dict(filename)

# %% load the model
tokenizer: FlairTokenizer = FlairTokenizer(letter_to_ix, embedding_dim, hidden_dim, num_layers, batch_size,
                                           use_CSE)
filename = 'test.tar'
tokenizer, optimizer = tokenizer._init_model_with_state_dict(filename)
# %%
error_sentence, results = tokenizer.evaluate([sentence])

# %% calculate loss for batch_size > 1
shuffle = True
batch_size = 10
train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=shuffle)
dev_loader = DataLoader(dataset=data_dev, batch_size=batch_size, shuffle=shuffle)

item = iter(train_loader).next()

from flair.data import LabeledString

data_points = LabeledString(item[0])
data_points.set_label('tokenization', item[1])
from flair.models.tokenizer_model import FlairTokenizer

# character_size = 10
embedding_dim = 4096
hidden_dim = 256
num_layers = 1
batch_size = 10
use_CSE = False
# init the tokenizer like you would your LSTMTagger
tokenizer: FlairTokenizer = FlairTokenizer(letter_to_ix, embedding_dim, hidden_dim, num_layers, batch_size,
                                           use_CSE)

# do a forward pass and compute the loss for the data point
tag_scores, loss = tokenizer.forward_loss(data_points)

# loss should be a single value tensor 
print(loss)
# %% the evaluate function does not work yet
error_sentence, results = tokenizer.evaluate(data_points)

# %%
state = tokenizer._get_state_dict()
tokenizer._init_model_with_state_dict(state)
#%%
state.keys()

#%%
def prepare_batch(data_points_str, to_ix):
    tensor_list = []
    for seq in data_points_str:
        idxs = [to_ix[w] for w in seq]
        tensor = torch.tensor(idxs, dtype=torch.long, device=flair.device)
        tensor_list.append(tensor)
    batch_tensor = pad_sequence(tensor_list, batch_first=False).squeeze()
    if len(batch_tensor.shape)==1:
        return batch_tensor.view(-1,1)
    else: return batch_tensor
    
#%%
from flair.models.tokenizer_model import *
x = prepare_batch(sentence.string,letter_to_ix)

x