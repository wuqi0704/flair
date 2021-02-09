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
                 letter_to_ix,
                 embedding_dim,
                 hidden_dim,
                 num_layers,
                 tagset_size,
                 batch_size,
                 use_CSE=False,
                 tag_to_ix = {'B': 0, 'I': 1,'E':2,'S':3,'X':4},
                 learning_rate = 0.1
                 ):

        super(FlairTokenizer, self).__init__()
        self.letter_to_ix = letter_to_ix
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.use_CSE = use_CSE
        self.tag_to_ix = tag_to_ix
        self.learning_rate = learning_rate

        self.character_embeddings = nn.Embedding(len(letter_to_ix), embedding_dim) 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=False, bidirectional=True)
        if self.use_CSE == True: 
            self.flair_embedding = FlairEmbeddings
            self.lm_f : LanguageModel = self.flair_embedding('multi-forward').lm
            self.lm_b : LanguageModel = self.flair_embedding('multi-backward').lm 
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
    
        self.loss_function = nn.NLLLoss()
    
    def prepare_cse(self,sentence,batch_size=1):
        if batch_size == 1:
            embeds_f = self.lm_f.get_representation([sentence],'\n','\n')[1:-1,:,:]
            embeds_b = self.lm_b.get_representation([sentence],'\n','\n')[1:-1,:,:]
        elif batch_size >1:
            embeds_f = self.lm_f.get_representation(list(sentence),'\n','\n')[1:-1,:,:]
            embeds_b = self.lm_b.get_representation(list(sentence),'\n','\n')[1:-1,:,:]
        return torch.cat((embeds_f,embeds_b),dim=2)
    
    def prepare_batch(self,data_points_str, to_ix):
        tensor_list = []
        for seq in data_points_str:
            idxs = [to_ix[w] for w in seq]
            tensor = torch.tensor(idxs, dtype=torch.long)
            tensor_list.append(tensor)
        return pad_sequence(tensor_list,batch_first=False).squeeze()# adjust dim for batch_size is one
    
    def find_token(self,sentence_str):
        token = []; word = ''
        for  i,tag in enumerate(sentence_str[1]):
            if tag == 'S':
                token.append(sentence_str[0][i])
                continue
            if tag == 'X': 
                continue 
            if (tag == 'B') | (tag == 'I'): 
                word += sentence_str[0][i] 
                continue
            if tag == 'E': 
                word+=sentence_str[0][i]
                token.append(word)
                word=''
        return token

    def prediction_str(self, input):
        import numpy as np
        ix_to_tag = {y:x for x,y in self.tag_to_ix.items()}
        output = [np.argmax(i) for i in input]
        out_list = [ix_to_tag[int(o)] for o in output]
        out_str = ''
        for o in out_list: out_str += o 
        return out_str

    @abstractmethod
    def forward_loss(
        self, data_points: Union[List[DataPoint], DataPoint]
    ) -> torch.tensor:
        """Performs a forward pass and returns a loss tensor for backpropagation. Implement this to enable training."""
        
        if self.use_CSE == True:
            embeds = self.prepare_cse(data_points.string,batch_size=self.batch_size).to(device)
        elif self.use_CSE == False:
            embeds = self.prepare_batch(data_points.string,self.letter_to_ix)
            embeds = self.character_embeddings(embeds) 

        x = embeds.view(embeds.shape[0], self.batch_size, -1)
        h0 = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        tag_space = self.hidden2tag(out.view(embeds.shape[0],self.batch_size, -1))
        tag_scores = F.log_softmax(tag_space, dim=2).squeeze() # dim = (len(data_points),batch,len(tag))
        
        tags = data_points.get_labels('tokenization')[0]._value
        targets = self.prepare_batch(tags,self.tag_to_ix).to(device=device)
        # if embeds.shape[1] != self.batch_size: continue
        if self.batch_size > 1: 
            length_list = []
            for sentence in data_points.string: length_list.append(len(sentence))
            tag_scores = pack_padded_sequence(tag_scores,length_list,enforce_sorted=False).data
            targets    = pack_padded_sequence(targets,length_list,enforce_sorted=False).data

        loss = self.loss_function(tag_scores,targets) 
        return tag_scores,loss

        # TODO: what is currently your forward() goes here, followed by the loss computation
        # Since the DataPoint brings its own label, you can compute the loss here

    @abstractmethod
    def evaluate(
        self,
        sentences: Union[List[DataPoint], Dataset],
        out_path: Path = None,
        embedding_storage_mode: str = "none",
    ) -> (Result, float):
        """Evaluates the model. Returns a Result object containing evaluation
        results and a loss value. Implement this to enable evaluation.
        :param data_loader: DataLoader that iterates over dataset to be evaluated
        :param out_path: Optional output path to store predictions
        :param embedding_storage_mode: One of 'none', 'cpu' or 'gpu'. 'none' means all embeddings are deleted and
        freshly recomputed, 'cpu' means all embeddings are stored on CPU, or 'gpu' means all embeddings are stored on GPU
        :return: Returns a Tuple consisting of a Result object and a loss float value
        """
        with torch.no_grad():
            import numpy as np; from tqdm import tqdm; import pandas as pd
            # load_checkpoint(torch.load(model_name,map_location=torch.device(embedding_storage_mode)), model, optimizer)
            # model,optimizer = self._init_model_with_state_dict()
            # state = self._get_state_dict()
            # self.load_state_dict(state['state_dict'])
            error_sentence = []; R_score,P_score,F1_score = [],[],[]
                
            for data_points in tqdm(sentences,position=0):
                # inputs = self.prepare_batch(sent,self.letter_to_ix)
                # print(model)
                # tag_scores = model(inputs)
                sent = data_points.string
                tag = data_points.get_labels('tokenization')[0]._value
                self.batch_size = 1
                tag_scores,loss = self.forward_loss(data_points)
                tag_predict = self.prediction_str(tag_scores)

                reference = self.find_token((sent,tag))
                candidate = self.find_token((sent,tag_predict))

                inter = [c for c in candidate if c in reference]
                if len(candidate) !=0: 
                    R = len(inter) / len(reference) 
                    P = len(inter) / len(candidate)
                else: 
                    R,P = 0,0 # when len(candidate) = 0, which means the model fail to extract any token from the sentence
                    error_sentence.append((sent,tag,tag_predict))
                
                if (len(candidate) !=0) & ((R+P)  != 0) : # if R = P = 0, meaning len(inter) = 0, R+P = 0
                    F1 = 2 * R*P / (R+P)
                else: 
                    F1=0 
                    if (sent,tag,tag_predict) not in error_sentence:
                        error_sentence.append((sent,tag,tag_predict))
                R_score.append(R); P_score.append(P);F1_score.append(F1)

            #['Recall','Precision','F1 score']
            results = (np.mean(R_score), np.mean(P_score),np.mean(F1_score))
            
            return error_sentence,results     
            
        # TODO: Your evaluation routine goes here. For the DataPoints passed into this method, compute the accuracy
        # and store it in a Result object, which you return.

    @abstractmethod
    def _get_state_dict(self,file=None):
        """Returns the state dictionary for this model. Implementing this enables the save() and save_checkpoint()
        functionality."""
        optimizer = optim.SGD(self.parameters(), self.learning_rate)
        checkpoint = {'state_dict' : self.state_dict(), 'optimizer': optimizer.state_dict()}
        if file != None:
            print("=> Saving checkpoint to: %s"%file)
            torch.save(checkpoint, file)
        return checkpoint

    # @staticmethod
    @abstractmethod # note: not so sure about this function
    def _init_model_with_state_dict(self,file=None):
        """Initialize the model from a state dictionary. Implementing this enables the load() and load_checkpoint()
        functionality."""
        optimizer = optim.SGD(self.parameters(), self.learning_rate)
        print("=> Loading checkpoint from : %s "%file)
        state = torch.load(file)
        # state = self._get_state_dict()
        self.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        return self,optimizer


    @staticmethod
    @abstractmethod
    def _fetch_model(model_name) -> str:
        return model_name

