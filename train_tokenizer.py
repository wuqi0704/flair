#%%
from flair.data import Corpus, LabeledString

from flair.datasets import SentenceDataset
from flair.embeddings import token
from flair.models.tokenizer_model import FlairTokenizer

# only one language for now
LanguageList = [
    'ENGLISH'
]
import pickle

# 1. load your data and convert to list of LabeledString
data_train, data_test, data_dev = [], [], []
for language in LanguageList:
    with open('resources/%s_Train.pickle' % language, 'rb') as f1:
        train = pickle.load(f1)
    with open('resources/%s_Test.pickle' % language, 'rb') as f2:
        test = pickle.load(f2)
    with open('resources/%s_Dev.pickle' % language, 'rb') as f3:
        dev = pickle.load(f3)

    data_train.extend([LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in train])
    data_test.extend([LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in test])
    data_dev.extend([LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in dev])


#%%
# 2. make a Corpus object
corpus: Corpus = Corpus(SentenceDataset(data_train), SentenceDataset(data_test), SentenceDataset(data_dev))
corpus = corpus.downsample(0.01)
# 3. make the letter dictionary from the corpus
letter_to_ix = {}
letter_to_ix[''] = 0  # need this for padding

for sentence in corpus.get_all_sentences():
    for letter in sentence.string:
        if letter not in letter_to_ix:
            letter_to_ix[letter] = len(letter_to_ix)
print('functions.py : Nr. of distinguish character: ', len(letter_to_ix.keys()))

# 4. initialize tokenizer
tokenizer: FlairTokenizer = FlairTokenizer(
    letter_to_ix=letter_to_ix,
    embedding_dim=4096,
    hidden_dim=256,
    num_layers=1,
    use_CSE=False
)

# 5. initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tokenizer, corpus)
#%%
# 6. train
trainer.train(
    "resources/taggers/example-tokenizer",
    learning_rate=0.1,
    mini_batch_size=1,
    max_epochs=5,
)

#%%
corpus.dev