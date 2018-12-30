import os
import pandas as pd
import numpy as np
import re
import json

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

## some config values 
embed_size = 300 # how big is each word vector
max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70 # max number of words in a question to use

DATA_DIR = './data_in/'
TRAIN_DATA = 'train.npy'
TEST_DATA = 'test.npy'
TEST_ID_DATA = 'test_id.npy'
TRAIN_LABEL_DATA = 'train_label.npy'
DATA_CONFIGS = 'data_configs.json'

def load_and_prec():
    train_df = pd.read_csv("./data_in/train.csv")
    test_df = pd.read_csv("./data_in/test.csv")
    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)
    
    ## fill up the missing values
    train_X = train_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values

    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    ## Pad the sentences 
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    ## Get the target values
    train_y = train_df['target'].values
    
    #shuffling the data
    np.random.seed(2018)
    trn_idx = np.random.permutation(len(train_X))

    train_X = train_X[trn_idx]
    train_y = train_y[trn_idx]
    
    return train_X, test_X, train_y, tokenizer.word_index

def load_glove(word_index):
    EMBEDDING_FILE = './data_in/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 
    
def load_fasttext(word_index):    
    EMBEDDING_FILE = './data_in/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix

def load_para(word_index):
    EMBEDDING_FILE = './data_in/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

if __name__ == '__main__':
    
    # For loading train & test dataset
    if os.path.exists(DATA_DIR + TRAIN_DATA):
        print("test data exists")
        input_data = np.load(open(DATA_DIR + TRAIN_DATA, 'rb'))
        label_data = np.load(open(DATA_DIR + TRAIN_LABEL_DATA, 'rb'))
        data_configs = json.load(open(DATA_DIR + DATA_CONFIGS, 'r'))
        word_vocab = data_configs['vocab']
    else:
        print("no preprocessing file exists")
        train_X, test_X, train_y, word_vocab = load_and_prec()
            
        data_configs = {}
        data_configs['vocab'] = word_vocab
        data_configs['vocab_size'] = len(word_vocab) + 1

        json.dump(data_configs, open(DATA_DIR + DATA_CONFIGS, 'w'))
        np.save(open(DATA_DIR + TEST_DATA, 'wb'), test_X)

        #save train and test dataset into numpy
        # np.save(open(DATA_DIR + TRAIN_DATA, 'wb'), train_X)
        # np.save(open(DATA_DIR + TRAIN_LABEL_DATA , 'wb'), train_y)
    
    print(input_data.shape)
    
    embedding_matrix_1 = load_glove(word_vocab)
    np.save(open(DATA_DIR + 'glove.npy', 'wb'), embedding_matrix_1)
    
    embedding_matrix_2 = load_fasttext(word_vocab)
    np.save(open(DATA_DIR + 'fasttext.npy', 'wb'), embedding_matrix_2)
    
    embedding_matrix_3 = load_para(word_vocab)
    np.save(open(DATA_DIR + 'para.npy', 'wb'), embedding_matrix_3)
    
    # np.save(open(DATA_DIR + 'glove.npy', 'wb'), embedding_matrix_1)

    # train_X, test_X, train_y, word_vocab = load_and_prec()

    # embedding_matrix_1 = load_glove(word_vocab)
    # embedding_matrix_2 = load_fasttext(word_vocab)
    # embedding_matrix_3 = load_para(word_vocab)

    