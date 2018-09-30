import os
from datetime import datetime
import json
from sklearn.model_selection import train_test_split
import numpy as np
import sys

import tensorflow as tf

import datetime, time, json
from tensorflow.python.keras._impl.keras.models import Model
from tensorflow.python.keras._impl.keras.layers import Input, Bidirectional, LSTM, dot, Flatten, Dense, Reshape, add, Dropout, BatchNormalization
from tensorflow.python.keras._impl.keras.layers.embeddings import Embedding
from tensorflow.python.keras._impl.keras.regularizers import l2
from tensorflow.python.keras._impl.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.python.keras._impl.keras import backend as K
from sklearn.model_selection import train_test_split


from tensorflow.python.keras._impl.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras._impl.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras._impl.keras.utils.data_utils import get_file


from tensorflow.python.keras._impl.keras import layers

# from tensorflow.python.keras._impl.keras.layers import Embedding
# from tensorflow.python.keras._impl.keras.layers import Reshape, Flatten, Dropout, Concatenate, dot, add

## 미리 Global 변수를 지정하자. 파일 명, 파일 위치, 디렉토리 등이 있다.

Q1_TRAINING_DATA_FILE = 'q1_train.npy'
Q2_TRAINING_DATA_FILE = 'q2_train.npy'
LABEL_TRAINING_DATA_FILE = 'label_train.npy'
WORD_EMBEDDING_MATRIX_FILE = 'word_embedding_matrix.npy'
NB_WORDS_DATA_FILE = 'nb_words.json'
DATA_PATH = './data/'
MODEL_WEIGHTS_FILE = './data/checkpoint/question_pairs_weights.h5'


## 학습에 필요한 파라메터들에 대해서 지정하는 부분이다.

BATCH_SIZE = 1024
EPOCH = 50
HIDDEN = 50
BUFFER_SIZE = 2048

TEST_SPLIT = 0.1
RNG_SEED = 13371447
WORD_EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 25
SENT_EMBEDDING_DIM = 50
DROPOUT = 0.2

VALIDATION_SPLIT = 0.2
NB_EPOCHS = 25

os.environ["CUDA_VISIBLE_DEVICES"]="4"

## 데이터를 불러오는 부분이다. 효과적인 데이터 불러오기를 위해, 미리 넘파이 형태로 저장시킨 데이터를 로드한다.

q1_data = np.load(open(DATA_PATH + Q1_TRAINING_DATA_FILE, 'rb'))
q2_data = np.load(open(DATA_PATH + Q2_TRAINING_DATA_FILE, 'rb'))
labels = np.load(open(DATA_PATH + LABEL_TRAINING_DATA_FILE, 'rb'))

## 미리 저장한 임베딩 값을 불러오자
word_embedding_matrix = np.load(open(DATA_PATH+WORD_EMBEDDING_MATRIX_FILE, 'rb'))
with open(DATA_PATH+NB_WORDS_DATA_FILE, 'r') as f:
    nb_words = json.load(f)['nb_words']

## 데이터를 나누어 저장하자. sklearn의 train_test_split을 사용하면 유용하다. 하지만, 쿼라 데이터의 경우는
## 입력이 1개가 아니라 2개이다. 따라서, np.stack을 사용하여 두개를 하나로 쌓은다음 활용하여 분류한다.

X = np.stack((q1_data, q2_data), axis=1) 
y = labels

# X = word_embedding_matrix[X[:50000]]
# y = y[:50000]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RNG_SEED)
Q1_train = X_train[:,0]
Q2_train = X_train[:,1]
Q1_test = X_test[:,0]
Q2_test = X_test[:,1]


question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

print(question1)

q1 = Embedding(nb_words + 1, 
                 WORD_EMBEDDING_DIM, 
                 weights=[word_embedding_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=False)(question1)
q1 = Bidirectional(LSTM(SENT_EMBEDDING_DIM, return_sequences=True), merge_mode="sum")(q1)

q2 = Embedding(nb_words + 1, 
                 WORD_EMBEDDING_DIM, 
                 weights=[word_embedding_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=False)(question2)
q2 = Bidirectional(LSTM(SENT_EMBEDDING_DIM, return_sequences=True), merge_mode="sum")(q2)

attention = dot([q1,q2], [1,1])
attention = Flatten()(attention)
attention = Dense((MAX_SEQUENCE_LENGTH*SENT_EMBEDDING_DIM))(attention)
attention = Reshape((MAX_SEQUENCE_LENGTH, SENT_EMBEDDING_DIM))(attention)

merged = add([q1,attention])
merged = Flatten()(merged)
merged = Dense(200, activation='relu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)
merged = Dense(200, activation='relu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)
merged = Dense(200, activation='relu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)
merged = Dense(200, activation='relu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)

is_duplicate = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[question1,question2], outputs=is_duplicate)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Starting training at", datetime.datetime.now())
t0 = time.time()
callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]
history = model.fit([Q1_train, Q2_train],
                    y_train,
                    epochs=NB_EPOCHS,
                    validation_split=VALIDATION_SPLIT,
                    verbose=1,  
                    batch_size=BATCH_SIZE,
                    callbacks=callbacks)
t1 = time.time()
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

