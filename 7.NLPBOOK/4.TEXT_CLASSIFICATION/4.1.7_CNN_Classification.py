import sys
import os
import string
import tempfile
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import json
from functools import partial

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

from tensorboard import summary as summary_lib

FILE_DIR_PATH = './data_in/'
INPUT_TRAIN_DATA_FILE_NAME = 'train_input.npy'
LABEL_TRAIN_DATA_FILE_NAME = 'train_label.npy'
INPUT_TEST_DATA_FILE_NAME = 'test_input.npy'

DATA_CONFIGS_FILE_NAME = 'data_configs.json'

train_input_data = np.load(open(FILE_DIR_PATH + INPUT_TRAIN_DATA_FILE_NAME, 'rb'))
train_label_data = np.load(open(FILE_DIR_PATH + LABEL_TRAIN_DATA_FILE_NAME, 'rb'))
test_input_data = np.load(open(FILE_DIR_PATH + INPUT_TEST_DATA_FILE_NAME, 'rb'))

prepro_configs = None

with open(FILE_DIR_PATH + DATA_CONFIGS_FILE_NAME, 'r') as f:
    prepro_configs = json.load(f)

# 파라메터 변수
RNG_SEED = 1234
BATCH_SIZE = 128
NUM_EPOCHS = 10
VOCAB_SIZE = len(prepro_configs)
EMB_SIZE = 128
VALID_SPLIT = 0.2
MAX_SEQ_LEN = 604 # 문장 최대 길이

input_train, input_valid, label_train, label_valid = train_test_split(train_input_data, train_label_data, test_size=VALID_SPLIT, random_state=RNG_SEED)

#문장 길이 구하는 값, 전처리로 같은 길이를 맞추어 놔서 의미 없음
len_train = np.array([min(len(x), MAX_SEQ_LEN) for x in input_train])
len_valid = np.array([min(len(x), MAX_SEQ_LEN) for x in input_valid])

def input_fn(X, y=None, is_training=False):

    def internal_input_fn(X, y=None, is_training=False):
        
        if (not isinstance(X, dict)):
            X = {"x": X}
        
        if (y is None):
            dataset = tf.data.Datset.from_tensor_slices(X)
        else:
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if (is_training):
            dataset = dataset.repeat().shuffle(len(X['x']))
            batch_size = BATCH_SIZE
        else:
            batch_size = 1

        dataset = dataset.batch(batch_size)
        dataset_iter = dataset.make_initializable_iterator()

        if (y is None):
            features = dataset_iter.get_next()
            labels = None
        else:
            features, labels = dataset_iter.get_next()

        input_tensor_map = dict()
        for input_name, tensor in features.items():
            input_tensor_map[input_name] = tensor.name

#         with open(os.path.join(my_dir, 'input_tensor_map.pickle'), 'wb') as f:
#             pickle.dump(input_tensor_map, f, protocol=pickle.HIGHEST_PROTOCOL)
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, dataset_iter.initializer)
        
        return (features, labels) if (not labels is None) else features
    
    return partial(internal_input_fn, X=X, y=y, is_training=is_training)

# input_fn(input_train, label_train)

def model_fn(features, labels, mode, params):

    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT
    
    #embedding layer를 선언합니다.
    input_layer = tf.contrib.layers.embed_sequence(
                    features['x'],
                    VOCAB_SIZE,
                    EMB_SIZE,
                    initializer=params['embedding_initializer']
                    )
    # 현재 모델이 학습모드인지 여부를 확인하는 변수입니다.
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    # embedding layer에 대한 output에 대해 dropout을 취합니다.
    dropout_emb = tf.layers.dropout(inputs=input_layer,
                                   rate=0.5,
                                   training=training)

    ## filters = 32이고 kernel_size = 3이면, 길이가 3인 32개의 다른 필터를 생성합니다. 32개의 컨볼루션들을 생성합니다.
    ## conv1d는 (배치사이즈, 길이, 채널)로 입력값을 받는데, 배치사이즈: 문장 숫자 | 길이: 각 문장의 단어의 개수 | 채널: 임베딩 출력 차원수임
    conv = tf.layers.conv1d(
            inputs=dropout_emb,
            filters=32,
            kernel_size=3,
            padding='same',
            activation=tf.nn.relu)
    
    pool = tf.reduce_max(input_tensor=conv, axis=1)
    hidden = tf.layers.dense(inputs=pool, units=250, activation=tf.nn.relu)
    dropout_hidden = tf.layers.dropout(inputs=hidden, rate=0.2, training=training)
    logits = tf.layers.dense(inputs=dropout_hidden, units=1)
    
    #prediction 진행 시, None
    if labels is not None:
        labels = tf.reshape(labels, [-1, 1])

    if TRAIN:
        global_step = tf.train.get_global_step()
        loss = tf.losses.sigmoid_cross_entropy(labels, logits)
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss, global_step)

        return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss = loss)
    
    elif EVAL:
        loss = tf.losses.sigmoid_cross_entropy(labels, logits)
        pred = tf.nn.sigmoid(logits)
        accuracy = tf.metrics.accuracy(labels, tf.round(pred))
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={'acc': accuracy})
        
    elif PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'prob': tf.nn.sigmoid(logits),
            }
        )

params = {'embedding_initializer': tf.random_uniform_initializer(-1.0, 1.0)}

model_dir = os.path.join(os.getcwd(), "data_out/checkpoint/cnn/")
os.makedirs(model_dir, exist_ok=True)

config_tf = tf.estimator.RunConfig()
config_tf._save_checkpoints_steps = 100
config_tf._save_checkpoints_secs = None
config_tf._keep_checkpoint_max =  2
config_tf._log_step_count_steps = 100

cnn_est = tf.estimator.Estimator(model_fn, model_dir=model_dir, config=config_tf, params=params)

for epoch in range(NUM_EPOCHS):
    print("Epoch : ", epoch)
    tf.logging.set_verbosity("INFO")

    # cnn_est.train(input_fn(X=input_train, y=label_train, is_training=True))
    eval_results = cnn_est.evaluate(input_fn(X=input_valid, y=label_valid, is_training=True))
    print("eval result:")
    print(eval_results)