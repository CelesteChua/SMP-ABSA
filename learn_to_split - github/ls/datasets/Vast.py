import os
import sys
import pandas as pd

import torch
import torch.nn.functional as F
import torchtext

import transformers
from transformers import BertTokenizer

from torch.utils.data import Dataset

class Vast(Dataset):
    def __init__(self):

        # get the data and binary targets
        self.data, self.labels = Vast.load_csv('ls/datasets/vast_data/vast_train.csv', 'ls/datasets/vast_data/vast_test.csv')


        self.length = len(self.labels)


    def load_csv(filename1, filename2):
        '''
            Load the csv files.
        '''
        
        data_1 = pd.read_csv(filename1)
        data_2 = pd.read_csv(filename2)
        data = pd.concat([data_1, data_2])
        
        
        PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        
        text = list(data.loc[:, "post"].array)
        targets = list(data.loc[:, "new_topic"].array)
        sentiment = data.loc[:, "label"].array

        #concat and token and tensor all tgt        # padding to ensure all same length [total 400]
        outputs = tokenizer(text, targets, max_length = 400, padding = "max_length")

        # output keys --> input_ids, attention_mask and token_type_ids 
        # changing from list to tensor ( int [long])
        input_ids = torch.tensor(outputs["input_ids"], dtype=torch.long)  #tensored 
        input_masks = torch.tensor(outputs["attention_mask"], dtype = torch.long) #which one has things 
        segment_ids = torch.tensor(outputs["token_type_ids"], dtype = torch.long) #tells code which one is which list

        #concat all to one variable 
        data = torch.stack((input_ids, input_masks, segment_ids), dim=2)  #following order in bert , dim=2 means it takes the first list first ele and put all tgt   


        labels= torch.tensor(sentiment, dtype = torch.long)

        return data, labels
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


   
  