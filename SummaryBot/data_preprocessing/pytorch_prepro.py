import os
import json
import requests
import re
import time

import pandas as pd
from pprint import pprint
from itertools import chain

from config import Config
from lib import data_utils

from torchtext.vocab import Vocab
from torchtext import data

config = Config()

data_path = config.data_path
kin_train = data_path + 'valid_pos_neg.txt'

train_query_idx = data_path + config.train_query_idx
train_pos_idx = data_path + config.train_pos_idx
train_neg_idx = data_path + config.train_neg_idx
train_vocab_idx = data_path + config.train_vocab_idx

######1. Data Loading#####
train_db = pd.read_csv(kin_train, sep="\t", header=None, names=["query", "pos", "neg1", "neg2", "neg3", "neg4"])

query = list(train_db['query'])
pos = list(train_db['pos'])
neg1 = list(train_db['neg1'])

QA_list = list(zip(query, pos, neg1))

#Data Loading

######2. Convert to Train#####
text_field = data.Field(
    use_vocab=True, #이미 ids로 치환여부
    fix_length=30, #길이 조정
    init_token="<SOS>",
    eos_token="<EOS>",
    pad_token="<PAD>",
    unk_token="<UNK>"
    )


examples = []
for q, p, n_1 in QA_list:
    example = data.Example.fromlist(
    data=[q, p, n_1],
    fields=[('query', text_field),
            ('pos', text_field),
           ('neg_1', text_field)])
    examples.append(example)

QA_dataset = data.Dataset(
    examples=examples,
    fields=[
            ('query', text_field),
            ('pos', text_field),
           ('neg_1', text_field)
    ])

#convert and save data into txt file

def data2list(dataset):
    train_query = []
    train_pos = []
    train_neg = []
    
    for i in range(len(dataset.examples)):
        train_query.append(vars(dataset.examples[i])['query'])
        train_pos.append(vars(dataset.examples[i])['pos'])
        train_neg.append(vars(dataset.examples[i])['neg_1'])
        
    return train_query, train_pos, train_neg

def list2txt(file_nm, db_list):
    with open(file_nm, 'w', encoding="utf-8") as f:
        for w in db_list:
            #print(w)
            f.write(' '.join(w) + "\n")

tr_query, tr_pos, tr_neg = data2list(QA_dataset)
            
list2txt('./data/kin/train_query.txt', tr_query)
list2txt('./data/kin/train_pos.txt', tr_pos)
list2txt('./data/kin/train_neg.txt', tr_neg)

#######3. Convert Vocab#########

# Build Vocab
text_field.build_vocab(QA_dataset)
vocab = text_field.vocab
len(vocab)

# Save vocab file
vocab_dict = dict(vocab.stoi)
with open(train_vocab_idx, 'w', encoding="utf-8") as f:
    f.write("\n".join("/".join(chain([str(key), str(value)])) for key,value in vocab_dict.items()))

#######4. Loading Data#######
def ids2token(ids):
    return [vocab.itos[id] for id in ids]

#데이터를 로드하여 index로 변환
#index변환 데이터를 Variable로 변환
query_train_path = data_path+'train_query.txt'
pos_train_path = data_path+'train_pos.txt'
neg_train_path = data_path+'train_neg.txt'

def sent2idx(sent_path, save_path, vocab_dict):
    with open(sent_path, "r", encoding="utf-8") as f:

        idx_sent = []
        read_lines = f.readlines()

        for line in read_lines:
            split_line = line.split("\n")
            sent = split_line[0].split(" ")
            #print(sent)

            # 한 줄의 질의에 대해 형태소 사전에 기재된 인덱스로 토큰을 치환한다
            u_token_idx_line = []
            for u_token in sent:
                try:
                    u_token_idx = vocab_dict[u_token]
                except KeyError:
                    u_token_idx = vocab_dict["<UNK>"]
                u_token_idx_line.append(str(u_token_idx))

            # 질의 끝에 EOS 아이디 붙인다
            u_token_idx_line.append(str(vocab_dict['<EOS>']))

            # 질의 한 줄씩 모은다
            idx_sent.append(u_token_idx_line)
            
        #파일을 저장한다.
        with open(save_path, "w", encoding="utf-8") as train_set:
            for i in range(len(idx_sent)):
                train_set.writelines(" ".join(idx_sent[i]) + "\n")

        return idx_sent
    
df_train_query = sent2idx(query_train_path, train_query_idx, vocab_dict)
df_train_pos = sent2idx(pos_train_path, train_pos_idx, vocab_dict)
df_train_neg = sent2idx(neg_train_path, train_neg_idx, vocab_dict)

print("Converting DB complete")

#데이터 불러오기

def load_idx(path, bucket=30):
    
    idx_sent = []
    
    with open(path, "r", encoding="utf-8") as f:

        read_lines = f.readlines()
        for line in read_lines:
            split_line = line.split("\n")
            sent = ([int(token) for token in split_line[0].split(" ")] + [(vocab_dict["<PAD>"])] * bucket)[:bucket]
            idx_sent.append(sent)
            
    return idx_sent

def vocab_size():
    #임시용..
    
    vocab_len = len(vocab)
    
    return vocab_len

print(vocab_size())
