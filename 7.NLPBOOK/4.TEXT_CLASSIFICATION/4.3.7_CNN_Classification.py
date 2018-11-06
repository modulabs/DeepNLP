import sys
import os
import numpy as np
import json

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, Bidirectional
from tensorflow.keras import backend as K


os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf.logging.set_verbosity("INFO")

DEFAULT_PATH ='~/.kaggle/competitions/word2vec-nlp-tutorial/'
DATA_IN_PATH = './data_in/'
DATA_OUT_PATH = './data_out/'
INPUT_TRAIN_DATA_FILE_NAME = 'train_input.npy'
LABEL_TRAIN_DATA_FILE_NAME = 'train_label.npy'
INPUT_TEST_DATA_FILE_NAME = 'test_input.npy'

DATA_CONFIGS_FILE_NAME = 'data_configs.json'

train_input_data = np.load(open(DATA_IN_PATH + INPUT_TRAIN_DATA_FILE_NAME, 'rb'))
train_label_data = np.load(open(DATA_IN_PATH + LABEL_TRAIN_DATA_FILE_NAME, 'rb'))
test_input_data = np.load(open(DATA_IN_PATH + INPUT_TEST_DATA_FILE_NAME, 'rb'))

prepro_configs = None

with open(DATA_IN_PATH + DATA_CONFIGS_FILE_NAME, 'r') as f:
    prepro_configs = json.load(f)

# 파라메터 변수
RNG_SEED = 1234
BATCH_SIZE = 128
NUM_EPOCHS = 20005
VOCAB_SIZE = len(prepro_configs)
EMB_SIZE = 128
VALID_SPLIT = 0.2
MAX_SEQ_LEN = 604 # 문장 최대 길이

train_input, eval_input, train_label, eval_label = train_test_split(train_input_data, train_label_data, test_size=VALID_SPLIT, random_state=RNG_SEED)

def mapping_fn(X, Y=None):
    input, label = {'x': X}, Y
    return input, label

def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((train_input, train_label))
    dataset = dataset.shuffle(buffer_size=len(train_input))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(mapping_fn)
    dataset = dataset.repeat(count=NUM_EPOCHS)

    iterator = dataset.make_one_shot_iterator()
    
    return iterator.get_next()

def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((eval_input, eval_label))
    dataset = dataset.shuffle(buffer_size=len(eval_input))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(mapping_fn)

    iterator = dataset.make_one_shot_iterator()
    
    return iterator.get_next()

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

    # dropout_emb = layers.Dropout(rate=0.5)(input_layer)
    dropout_emb = tf.layers.dropout(inputs=input_layer,
                                   rate=0.5,
                                   training=training)

    with tf.variable_scope('lstm'):
        # lstm_layers = Bidirectional(layers.CuDNNLSTM(32, return_sequences=True, dropout=0.5))
        lstm_layers = Bidirectional(layers.CuDNNLSTM(64, return_sequences=True))
        lstm = lstm_layers(dropout_emb)

    with tf.variable_scope('cnn'):
        cnns = [layers.Conv1D(kernel_size=kernel_size, filters=500, activation='tanh', padding='same') for kernel_size in [2,3,5]]
        cnn = layers.concatenate([layers.GlobalMaxPooling1D()(cnn(lstm)) for cnn in cnns], axis=-1)
        
    with tf.variable_scope('output_layer'):
        hidden1 = tf.layers.dense(inputs=cnn, units=128, activation=tf.nn.relu)
        dropout = layers.Dropout(0.5)(hidden1)
        logits = layers.Dense(1, name='logits')(dropout)
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
                'prob': tf.nn.sigmoid(logits)
            }
        )


params = {'embedding_initializer': tf.random_uniform_initializer(-1.0, 1.0)}

model_dir = os.path.join(os.getcwd(), "data_out/checkpoint/rnn/")
os.makedirs(model_dir, exist_ok=True)

config_tf = tf.estimator.RunConfig()
# config_tf._save_checkpoints_secs = 100
config_tf._keep_checkpoint_max =  2
config_tf._log_step_count_steps = 100

cnn_est = tf.estimator.Estimator(model_fn, model_dir=model_dir, config=config_tf, params=params)

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=NUM_EPOCHS)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=0)

tf.estimator.train_and_evaluate(cnn_est, train_spec, eval_spec)

# cnn_est = tf.estimator.Estimator(model_fn, model_dir=model_dir, config=config_tf, params=params)
# cnn_est.train(train_input_fn) #학습하기
# cnn_est.evaluate(eval_input_fn) #평가하기

#Model1: INFO:tensorflow:Saving dict for global step 5000: acc = 0.8208, global_step = 5000, loss = 0.9476856
#Model2: INFO:tensorflow:Saving dict for global step 5000: acc = 0.8872, global_step = 5000, loss = 0.9010733
#Model3: INFO:tensorflow:Saving dict for global step 5000: acc = 0.8718, global_step = 5000, loss = 0.43977556
#Model4: INFO:tensorflow:Saving dict for global step 5000: acc = 0.8742, global_step = 5000, loss = 0.450572
#model4(LSTM): INFO:tensorflow:Saving dict for global step 20005: acc = 0.8832, global_step = 20005, loss = 1.4345698

# 예측된 모델을 불러 체크포인트로 결과치를 불러온다.
test_input_data = np.load(open(DATA_IN_PATH + INPUT_TEST_DATA_FILE_NAME, 'rb')) #테스트데이터 로드

predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":test_input_data}, shuffle=False) #numpy 형태로 저장
cnn_est.predict(input_fn=predict_input_fn)

predictions = np.array([p['prob'][0] for p in cnn_est.predict(input_fn=predict_input_fn)])

import pandas as pd
#테스트 데이터 로드
test = pd.read_csv(DEFAULT_PATH+"testData.tsv", header=0, delimiter="\t", quoting=3 )

print ("test dataset shape: {}".format(test.shape))

#알아보기 쉽게 데이터랑 붙여두는 편이 좋을 거 같습니다.
output = pd.DataFrame( data={"id":test["id"], "sentiment":list(predictions)} )

#지금까지 처리한 결과를 파일로 저장합니다.
output.to_csv("./data_out/Bag_of_Words_model_test.csv", index=False, quoting=3 )