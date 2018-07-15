import os
from datetime import datetime
import tensorflow as tf

import numpy as np
import json

FILE_DIR_PATH = './data/'
INPUT_TRAIN_DATA_FILE_NAME = 'input.npy'
LABEL_TRAIN_DATA_FILE_NAME = 'label.npy'
DATA_CONFIGS_FILE_NAME = 'data_configs.json'

input_data = np.load(open(FILE_DIR_PATH + INPUT_TRAIN_DATA_FILE_NAME, 'rb'))
label_data = np.load(open(FILE_DIR_PATH + LABEL_TRAIN_DATA_FILE_NAME, 'rb'))
prepro_configs = None

from sklearn.model_selection import train_test_split

TEST_SPLIT = 0.1
RNG_SEED = 13371447

input_train, input_test, label_train, label_test = train_test_split(input_data, label_data, test_size=TEST_SPLIT, random_state=RNG_SEED)

BATCH_SIZE = 16
NUM_EPOCHS = 10
vocab_size = 74065
embedding_size = 128

def mapping_fn(X, Y):
    input, label = {'text': X}, Y
    return input, label

def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((input_train, label_train))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(mapping_fn)
    dataset = dataset.repeat(count=NUM_EPOCHS)
    iterator = dataset.make_one_shot_iterator()
    
    return iterator.get_next()

def test_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((input_test, label_test))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(mapping_fn)
    iterator = dataset.make_one_shot_iterator()
    
    return iterator.get_next()

def model_fn(features, labels, mode, params):
    #embedding layer를 선언합니다.
    input_layer = tf.contrib.layers.embed_sequence(
                    features['text'],
                    vocab_size,
                    embedding_size,
                    initializer=params['embedding_initializer']
                    )
    # 현재 모델이 학습모드인지 여부를 확인하는 변수입니다.
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    # embedding layer에 대한 output에 대해 dropout을 취합니다.
    dropout_emb = tf.layers.dropout(inputs=input_layer,
                                   rate=0.2,
                                   training=training)
    
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
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                  mode=mode,
                  predictions={
                      'prob':tf.nn.sigmoid(logits)
                  })
    else:
        global_step = tf.train.get_global_step()
        loss = tf.losses.sigmoid_cross_entropy(labels, logits)
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss, global_step)
    
        return tf.estimator.EstimatorSpec(
                  mode=mode,
                  train_op=train_op,
                  loss=loss)

params = {'embedding_initializer': tf.random_uniform_initializer(-1.0, 1.0)}

model_dir = os.path.join(os.getcwd(), "checkpoint/cnn_model")
os.makedirs(model_dir, exist_ok=True)

config_tf = tf.estimator.RunConfig()
config_tf._save_checkpoints_steps = 100
config_tf._save_checkpoints_secs = None
config_tf._keep_checkpoint_max =  2
config_tf._log_step_count_steps = 100

tf.logging.set_verbosity(tf.logging.INFO)
print(tf.__version__)
time_start = datetime.utcnow()
print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
print(".......................................") 

est = tf.estimator.Estimator(model_fn, model_dir=model_dir, config=config_tf, params=params)
est.train(train_input_fn)

prediction = est.predict(test_input_fn)
# for i, p in enumerate(prediction):
#     print(i, p['q_sem'])

time_end = datetime.utcnow()
print(".......................................")
print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
print("")
time_elapsed = time_end - time_start
print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
