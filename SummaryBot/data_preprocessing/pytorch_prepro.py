import sys
import os
import json
import requests
import re
import time
import numpy as np

import pandas as pd
from pprint import pprint
from itertools import chain

# from config import Config
# from lib import data_utils

# label_path = "/Users/user/DeepNLP/SummaryBot/data/label/1801.08985"

# with open(label_path, 'r') as f:
#     print(f.readlines())

from torchtext.vocab import Vocab
from torchtext import data

import sys
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


abstract_path = '/Users/user/DeepNLP/SummaryBot/data_title/abstract/'
title_path = '/Users/user/DeepNLP/SummaryBot/data_title/title/'

#타이틀과 Abstract데이터를 로드한다. title은 /n을 공백으로 치환하고, abstract같은 경우에는 /n을 '\u2028' - line Seperator로 치환
#https://www.fileformat.info/info/unicode/char/2028/index.htm


vocab_dict={
    "<PAD>": 1
}

def load_title(path, bucket=10):
    
    idx_title = []

    with open(path, 'r') as f:

        read_lines = f.readlines()
        
        for line in read_lines:
            split_line = line.split("\n")
            sent = ([int(token) for token in split_line[0].split(" ")] + [(vocab_dict["<PAD>"])] * bucket)[:bucket]
            idx_title.append(sent)
    
    return idx_title

def load_abstract(path, bucket=10):
    
    idx_sent = []
    # idx_tot = []
    
    with open(path, "r", encoding="utf-8") as f:

        read_lines = f.readlines()
        for line in read_lines:
            split_line = line.split("\n")
            try:
                # idx_sent.append(split_line)
                sent = ([int(token) for token in split_line[0].split(" ")] + [(vocab_dict["<PAD>"])] * bucket)[:bucket]
                # sent = ([token for token in split_line[0].split(" ")] + [(vocab_dict["<PAD>"])] * bucket)[:bucket]
                idx_sent.append(sent)
            except Exception as e:
                pass
        
        idx_tot = list(filter(None, sum(idx_sent, [])))
        idx_tot = [item for item in idx_tot if item > 1]
    
    # idx_tot = sum(idx_sent, [])
                
    return idx_tot

df_abstract = Variable(torch.LongTensor(load_abstract(abstract_path+'1801.08618_ids.enc')))
df_title = Variable(torch.LongTensor(load_title(title_path+'1801.08618_idsa.dec')))

print(df_abstract)
print(df_title)

sys.exit(0)


# #배치 형태의 데이터 치환
# batch_size = 1

# batch_idx = np.random.choice(np.arange(len(df_title)), int(batch_size), False).astype(int)
# print(batch_idx)
