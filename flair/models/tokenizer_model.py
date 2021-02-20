import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Union, List

import torch
from torch.utils.data.dataset import Dataset

import flair
from flair.data import DataPoint
from flair.training_utils import Result

import torch
import torch.nn as nn
import torch.nn.functional as F
from flair.embeddings import FlairEmbeddings
from flair.models import LanguageModel
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class FlairTokenizer(flair.nn.Model):

    def __init__(self,
                 letter_to_ix, # character dictionary 
                 embedding_dim=4096,
                 hidden_dim=256,
                 num_layers=1,
                 batch_size=1,
                 use_CSE=False,
                 tag_to_ix={'B': 0, 'I': 1, 'E': 2, 'S': 3, 'X': 4},
                 learning_rate=0.1,
                 use_CRF = False
                 ):

        super(FlairTokenizer, self).__init__()
        self.letter_to_ix = letter_to_ix
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.use_CSE = use_CSE
        self.tag_to_ix = tag_to_ix
        self.learning_rate = learning_rate
        self.use_CRF = use_CRF
        
        if self.use_CSE == False:
            self.character_embeddings = nn.Embedding(len(letter_to_ix), embedding_dim)
            self.embeddings = self.character_embeddings
        elif self.use_CSE == True:
            self.flair_embedding = FlairEmbeddings
            self.lm_f: LanguageModel = self.flair_embedding('multi-forward').lm
            self.lm_b: LanguageModel = self.flair_embedding('multi-backward').lm
            self.embeddings = self.flair_embedding

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=False, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, len(tag_to_ix))
        self.loss_function = nn.NLLLoss()
        if self.use_CRF == True:
            # Matrix of transition parameters.  Entry i,j is the score of
             # transitioning *to* i *from* j.
            self.transitions = nn.Parameter(
                torch.randn(len(self.tag_to_ix), len(self.tag_to_ix)))
            # These two statements enforce the constraint that we never transfer
            # to the start tag and we never transfer from the stop tag
            START_TAG = "<START>"
            STOP_TAG = "<STOP>"
            self.transitions.data[tag_to_ix[START_TAG], :] = -10000
            self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        
        self.hidden = self.init_hidden()
        
        self.to(flair.device)

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim ),
                torch.randn(2, 1, self.hidden_dim ))

    def prepare_cse(self, sentence, batch_size=1):
        if batch_size == 1:
            embeds_f = self.lm_f.get_representation([sentence], '\n', '\n')[1:-1, :, :]
            embeds_b = self.lm_b.get_representation([sentence], '\n', '\n')[1:-1, :, :]
        elif batch_size > 1:
            embeds_f = self.lm_f.get_representation(list(sentence), '\n', '\n')[1:-1, :, :]
            embeds_b = self.lm_b.get_representation(list(sentence), '\n', '\n')[1:-1, :, :]
        return torch.cat((embeds_f, embeds_b), dim=2)
    @staticmethod
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

    @staticmethod
    def find_token(sentence_str):
        token = []
        word = ''
        for i, tag in enumerate(sentence_str[1]):
            if tag == 'S':
                token.append(sentence_str[0][i])
                continue
            if tag == 'X':
                continue
            if (tag == 'B') | (tag == 'I'):
                word += sentence_str[0][i]
                continue
            if tag == 'E':
                word += sentence_str[0][i]
                token.append(word)
                word = ''
        return token

    def prediction_str(self, input):
        ix_to_tag = {y: x for x, y in self.tag_to_ix.items()}
        output = [torch.argmax(i) for i in input]
        out_list = [ix_to_tag[int(o)] for o in output]
        out_str = ''
        for o in out_list: out_str += o
        return out_str

    @abstractmethod
    def forward_loss(
            self, data_points: Union[List[DataPoint], DataPoint],
            foreval = False,
    ) -> torch.tensor:
        """Performs a forward pass and returns a loss tensor for backpropagation. Implement this to enable training."""
        # if (self.batch_size > 1) : 
        try:
            sent_string,tags = [],[]
            for sentence in data_points: 
                sent_string.append((sentence.string))
                tags.append(sentence.get_labels('tokenization')[0]._value)
            batch_size=len(data_points)
        except: 
            sent_string = data_points.string
            tags = data_points.get_labels('tokenization')[0]._value
            batch_size = 1
        
        targets = self.prepare_batch(tags, self.tag_to_ix).squeeze().to(device=device)
        if self.use_CSE == True:
            embeds = self.prepare_cse(sent_string, batch_size=batch_size).to(device)
        elif self.use_CSE == False:
            embeds = self.prepare_batch(sent_string, self.letter_to_ix)
            embeds = self.character_embeddings(embeds)

        
        h0 = torch.zeros(self.num_layers * 2, embeds.shape[1], self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers * 2, embeds.shape[1], self.hidden_dim).to(device)
        out, _ = self.lstm(embeds, (h0, c0))
        tag_space = self.hidden2tag(out.view(embeds.shape[0], embeds.shape[1], -1))
        tag_scores = F.log_softmax(tag_space, dim=2).squeeze()  # dim = (len(data_points),batch,len(tag))

        if (batch_size > 1) : # if the input is more than one datapoint
            length_list = []
            for sentence in data_points: 
                length_list.append(len(sentence.string))
            tag_scores = pack_padded_sequence(tag_scores, length_list, enforce_sorted=False).data
            targets = pack_padded_sequence(targets, length_list, enforce_sorted=False).data
        
        loss = self.loss_function(tag_scores, targets)
        if foreval: return tag_scores
        else: return loss

        # TODO: what is currently your forward() goes here, followed by the loss computation
        # Since the DataPoint brings its own label, you can compute the loss here
    
    
    @abstractmethod
    def evaluate(
            self,
            sentences: Union[List[DataPoint], Dataset],
            out_path: Path = None,
            embedding_storage_mode: str = "none",
            mini_batch_size: int = 32,
            num_workers: int = 8,
            wsd_evaluation: bool = False
    ) -> (Result, float):
        """Evaluates the model. Returns a Result object containing evaluation
        results and a loss value. Implement this to enable evaluation.
        :param data_loader: DataLoader that iterates over dataset to be evaluated
        :param out_path: Optional output path to store predictions
        :param embedding_storage_mode: One of 'none', 'cpu' or 'gpu'. 'none' means all embeddings are deleted and
        freshly recomputed, 'cpu' means all embeddings are stored on CPU, or 'gpu' means all embeddings are stored on GPU
        :return: Returns a Tuple consisting of a Result object and a loss float value
        """
        # print("evaluate batch_size : ",mini_batch_size)
        with torch.no_grad():
            import numpy as np;
            from tqdm import tqdm;
            error_sentence = []
            R_score, P_score, F1_score = [], [], []

            for data_points in tqdm(sentences, position=0):
                sent = data_points.string
                tag = data_points.get_labels('tokenization')[0]._value
                tag_scores = self.forward_loss(data_points,foreval=True) 
                # print(tag_scores)
                tag_predict = self.prediction_str(tag_scores)

                reference = self.find_token((sent, tag))
                candidate = self.find_token((sent, tag_predict))

                inter = [c for c in candidate if c in reference]
                if len(candidate) != 0:
                    R = len(inter) / len(reference)
                    P = len(inter) / len(candidate)
                else:
                    R, P = 0, 0  # when len(candidate) = 0, which means the model fail to extract any token from the sentence
                    error_sentence.append((sent, tag, tag_predict))

                if (len(candidate) != 0) & ((R + P) != 0):  # if R = P = 0, meaning len(inter) = 0, R+P = 0
                    F1 = 2 * R * P / (R + P)
                else:
                    F1 = 0
                    if (sent, tag, tag_predict) not in error_sentence:
                        error_sentence.append((sent, tag, tag_predict))
                R_score.append(R)
                P_score.append(P)
                F1_score.append(F1)



            # ['Recall','Precision','F1 score']
            results = (np.mean(R_score), np.mean(P_score), np.mean(F1_score))

            return error_sentence, results

            # TODO: Your evaluation routine goes here. For the DataPoints passed into this method, compute the accuracy
        # and store it in a Result object, which you return.


    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            'letter_to_ix': self.letter_to_ix,
            'embedding_dim' : self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers':self.num_layers,
            'batch_size':self.batch_size,
            'use_CSE':self.use_CSE,
            'tag_to_ix':self.tag_to_ix,
            'learning_rate':self.learning_rate,
            'use_CRF':self.use_CRF
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        model = FlairTokenizer(
            letter_to_ix = state['letter_to_ix'],
            embedding_dim = state['embedding_dim'],
            hidden_dim = state['hidden_dim'],
            num_layers = state['num_layers'],
            batch_size = state['batch_size'],
            use_CSE = state['use_CSE'],
            tag_to_ix = state['tag_to_ix'],
            learning_rate = state['learning_rate'],
            use_CRF = state['use_CRF']
        )
        model.load_state_dict(state["state_dict"])
        return model


    @staticmethod
    @abstractmethod
    def _fetch_model(model_name) -> str:
        return model_name
