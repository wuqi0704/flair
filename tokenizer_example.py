
#%%
from torch.utils.data import DataLoader
from typing import Union, List
LanguageList = [
    'HEBREW','ARABIC','PORTUGUESE','ITALIAN','FRENCH','SPANISH','GERMAN','ENGLISH',
    'RUSSIAN','FINNISH','VIETNAMESE','KOREAN','CHINESE','JAPANESE'
]
import pickle
data_train,data_test,data_dev=[],[],[]
for language in LanguageList:
    with open('/Users/wuqi/MasterThesis_Tokenization/data/%s_Train.pickle'%language, 'rb') as f1:
        train = pickle.load(f1)
    with open('/Users/wuqi/MasterThesis_Tokenization/data/%s_Test.pickle'%language, 'rb') as f2:
        test = pickle.load(f2) 
    with open('/Users/wuqi/MasterThesis_Tokenization/data/%s_Dev.pickle'%language, 'rb') as f3:
        dev = pickle.load(f3)
    
    data_train += train; data_test += test; data_dev += dev

letter_to_ix = {}
letter_to_ix[''] = 0 # need this for padding
for sent, tags in data_train+data_test+data_dev:
    for letter in sent:
        if letter not in letter_to_ix:
            letter_to_ix[letter] = len(letter_to_ix)
print('functions.py : Nr. of distinguish character: ',len(letter_to_ix.keys()))

tag_to_ix = {'B': 0, 'I': 1,'E':2,'S':3,'X':4}

shuffle = True
batch_size=10
train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=shuffle)
dev_loader = DataLoader(dataset=data_dev, batch_size=batch_size, shuffle=shuffle)


#%%
from flair.data import LabeledString
# import torch
# LabeledString is a DataPoint - init and set the label
sentence = LabeledString('Any major dischord and we all suffer.')
sentence.set_label('tokenization', 'BIEXBIIIEXBIIIIIIEXBIEXBEXBIEXBIIIIES')

# Print the DataPoint
print(sentence)

# Print the string
print(sentence.string)

# print the label
print(sentence.get_labels('tokenization'))


tag_to_ix = {'B': 0, 'I': 1,'E':2,'S':3,'X':4} 
ix_to_tag = {y:x for x,y in tag_to_ix.items()}

# from flair.models.tokenizer_model import *
from flair.models.tokenizer_model import FlairTokenizer
# character_size = 10
embedding_dim = 4096
hidden_dim = 256
num_layers = 1
tagset_size= 5
batch_size = 1
use_CSE = False
# init the tokenizer like you would your LSTMTagger
tokenizer: FlairTokenizer = FlairTokenizer(letter_to_ix, embedding_dim, hidden_dim, num_layers, tagset_size, batch_size,use_CSE)

# do a forward pass and compute the loss for the data point
tag_scores,loss = tokenizer.forward_loss(sentence)

# loss should be a single value tensor 
print(loss)
#%% save the model
filename = 'test.tar'
checkpoint = tokenizer._get_state_dict(filename)

#%% load the model 
tokenizer: FlairTokenizer = FlairTokenizer(letter_to_ix, embedding_dim, hidden_dim, num_layers, tagset_size, batch_size,use_CSE)
filename = 'test.tar'
tokenizer,optimizer = tokenizer._init_model_with_state_dict(filename)
#%%
error_sentence,results  = tokenizer.evaluate([sentence])

#%% calculate loss for batch_size > 1
shuffle = True
batch_size=10
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
tagset_size= 5
batch_size = 10
use_CSE = False
# init the tokenizer like you would your LSTMTagger
tokenizer: FlairTokenizer = FlairTokenizer(letter_to_ix, embedding_dim, hidden_dim, num_layers, tagset_size, batch_size,use_CSE)

# do a forward pass and compute the loss for the data point
tag_scores,loss = tokenizer.forward_loss(data_points)

# loss should be a single value tensor 
print(loss)
#%%
error_sentence,results  = tokenizer.evaluate(data_points)

#%%
data_points.string