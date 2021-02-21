# %%
# import flair
from flair.datasets import DataLoader
from flair.embeddings import token
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

# tag_to_ix = {'B': 0, 'I': 1, 'E': 2, 'S': 3, 'X': 4}
# ix_to_tag = {y: x for x, y in tag_to_ix.items()}

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
use_CSE = True
use_CRF = True
# init the tokenizer like you would your LSTMTagger
tokenizer: FlairTokenizer = FlairTokenizer(letter_to_ix, embedding_dim, hidden_dim, num_layers,
                                           use_CSE,use_CRF=use_CRF)

# FIXME: do a forward pass and compute the loss for two data points
print(tokenizer.forward_loss([sentence, sentence_2]))
print(tokenizer.forward_loss(sentence))


#%%
sentences = [sentence, sentence_2]
# tokenizer.forward_loss(sentences,foreval=True)
# tokenizer.forward_loss(sentence,foreval=True)
tokenizer.evaluate(sentences)
tokenizer.evaluate(sentence) # FIXME: system stop running if I try tokenizer.evaluate([sentence])
# TypeError: object of type 'LabeledString' has no len()


#%%
from flair.datasets import SentenceDataset
sentence = SentenceDataset(sentence)
# tokenizer.evaluate(sentence)

# %% load the model
tokenizer: FlairTokenizer = FlairTokenizer(letter_to_ix, embedding_dim, hidden_dim, num_layers,
                                           use_CSE)
filename = 'test.tar'
tokenizer, optimizer = tokenizer._init_model_with_state_dict(filename)


# %%
state = tokenizer._get_state_dict()
tokenizer._init_model_with_state_dict(state)