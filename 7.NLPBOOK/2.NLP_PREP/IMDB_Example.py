#!/usr/bin/env python3
#-*- coding: utf-8 -*-
​
import os
import tempfile
import tensorflow as tf
import sys
​
from tensorflow.python.keras.datasets import imdb #imdb dataset
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import utils
from tensorboard import summary as summary_lib
​
# import matplotlib.pyplot as plt
# import seaborn as sns
​
import numpy as np
​
tf.logging.set_verbosity(tf.logging.INFO)
print(tf.__version__)
​
os.environ["CUDA_VISIBLE_DEVICES"]="0" #For TEST
​
vocab_size = 5000
sentence_size = 200
embedding_size = 50
model_dir = tempfile.mkdtemp()
​
NUM_EPOCHS = 10
BATCH_SIZE = 128
HIDDEN_DIM = 32
KEEP_PROB = 0.3
​
pad_id = 0
start_id = 1
oov_id = 2
index_offset = 2
​
(x_train_variable, y_train), (x_test_variable, y_test) = imdb.load_data(num_words=vocab_size,
                           start_char=start_id,
                           oov_char=oov_id,
                           index_from=index_offset)
​
print("x_train shape", x_train_variable.shape)
print("x_test shape", x_test_variable.shape)
​
print("Pad sequences (samples x time)")
​
x_train = sequence.pad_sequences(x_train_variable, 
                 maxlen=sentence_size,
                 truncating='post',
                 padding='post',
                 value=pad_id)
x_test = sequence.pad_sequences(x_test_variable, 
                maxlen=sentence_size,
                truncating='post',
                padding='post', 
                value=pad_id)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
​
x_len_train = np.array([min(len(x), sentence_size) for x in x_train_variable])
x_len_test = np.array([min(len(x), sentence_size) for x in x_test_variable])
​
word_index = imdb.get_word_index()
word_inverted_index = {v + index_offset: k for k, v in word_index.items()}
​
# The first indexes in the map are reserved to represent things other than tokens
word_inverted_index[pad_id] = '<PAD>'
word_inverted_index[start_id] = '<START>'
word_inverted_index[oov_id] = '<OOV>'
​
# for i in range(0, 10):
#   print(i, word_inverted_index[i])
  
def index_to_text(indexes):
  return ' '.join([word_inverted_index[i] for i in indexes])
​
# print(index_to_text(x_train_variable[0]))
​
​
def parser(input_sent, length, label):
  features = {"input_sent": input_sent, "len": length}
  return features, label
​
#the former (repeat before shuffle) provides better performance, 
# while the latter (shuffle before repeat) provides stronger ordering guarantees.
​
def train_input_fn():
  with tf.device('/cpu:0'):
    dataset = tf.data.Dataset.from_tensor_slices((x_train, x_len_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(x_train))
    dataset = dataset.repeat()
    dataset = dataset.map(parser)
    dataset = dataset.batch(BATCH_SIZE)
​
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()
​
def eval_input_fn():
  with tf.device('/cpu:0'):
    dataset = tf.data.Dataset.from_tensor_slices((x_train, x_len_train, y_train))
    dataset = dataset.map(parser)
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()
​
def gru_classifier(features, labels, mode, params):
  
  TRAIN = mode == tf.estimator.ModeKeys.TRAIN
  EVAL = mode == tf.estimator.ModeKeys.EVAL
  PREDICT = mode == tf.estimator.ModeKeys.PREDICT
  
  input_layer = tf.contrib.layers.embed_sequence(
                        features['input_sent'],
                        vocab_size,
                        embedding_size,
                        initializer=params['embed_init']
                        )
  
  with tf.variable_scope('bi-directional_gru'):
    gru_fw_cell = tf.contrib.rnn.GRUCell(num_units = HIDDEN_DIM,
                           activation = tf.nn.tanh)
    gru_bw_cell = tf.contrib.rnn.GRUCell(num_units = HIDDEN_DIM,
                           activation = tf.nn.tanh)
    gru_fw_cell = tf.contrib.rnn.DropoutWrapper(gru_fw_cell, KEEP_PROB)
    gru_bw_cell = tf.contrib.rnn.DropoutWrapper(gru_bw_cell, KEEP_PROB)
​
    _, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = gru_fw_cell, cell_bw = gru_bw_cell,
                            inputs = input_layer, sequence_length = features['len'],
                            dtype = tf.float32)
    
    concat_gru_output = tf.concat([output_states[0], output_states[1]], axis = 1)
​
  with tf.variable_scope('output_layer'):
    score = tf.layers.dense(concat_gru_output, 1, name='score_layer')
      
  if labels is not None:
    # labels = tf.one_hot(labels,2)
    labels = tf.reshape(labels, [-1, 1])
  # print(labels)
  
  optimizer = tf.train.AdamOptimizer()
  
  if TRAIN:
    global_step = tf.train.get_global_step()
    loss = tf.losses.sigmoid_cross_entropy(labels, score)
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step)
    return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss = loss)
  
  elif EVAL:
    loss = tf.losses.sparse_softmax_cross_entropy(labels, score)
    pred = tf.nn.sigmoid(score)
    accuracy = tf.metrics.accuracy(labels, tf.round(pred))
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={'acc': accuracy})
    
  elif PREDICT:
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions={
        'prob': score,
      }
    )
​
​
params = {'embed_init': tf.random_uniform_initializer(-1.0, 1.0)}
​
# Create Checkpoint Folder
model_dir = os.path.join(os.getcwd(), "data_out/gru")
os.makedirs(model_dir, exist_ok=True)
​
# Set Config
config_tf = tf.estimator.RunConfig()
# config_tf._save_checkpoints_steps = 100
config_tf._keep_checkpoint_max = 2
# config_tf._log_step_count_steps = 100
​
gru_classifier = tf.estimator.Estimator(model_fn = gru_classifier,
                    model_dir='./data_out/gru',
                    params=params,
                    config=config_tf)
​
def train_and_evaluate(classifier):
  classifier.train(input_fn=train_input_fn, steps=NUM_EPOCHS)
  eval_result = classifier.evaluate(input_fn=eval_input_fn)
​
train_and_evaluate(gru_classifier)
​
# 예측된 모델을 불러 체크포인트로 결과치를 불러온다.
​
def get_predictions(estimator, input_fn):
  return [x["prob"][0] for x in estimator.predict(input_fn=input_fn)]
​
LABELS = ["negative", "positive"]
​
# predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"input_sent":train_input_fn}, shuffle=False)
# gru_classifier.predict(input_fn=predict_input_fn)
​
with tf.Graph().as_default():
  cm = tf.confusion_matrix(y_train, get_predictions(gru_classifier, train_input_fn))
​
  with tf.Session() as session:
    cm_out = session.run(cm)
​
cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]
​
sns.heatmap(cm_out, annot=True, xticklabels=LABELS, yticklabels=LABELS)
plt.xlabel("Predicted")
plt.ylabel("True")
​
# predictions = np.array([p['logits'][0] for p in cnn_classifier.predict(input_fn=predict_input_fn)])
​
# #   # Reset the graph to be able to reuse name scopes
# #   tf.reset_default_graph() 
# #   # Add a PR summary in addition to the summaries that the classifier writes
# #   pr = summary_lib.pr_curve('precision_recall', predictions=predictions, labels=y_test.astype(bool), num_thresholds=21)
# #   with tf.Session() as sess:
# #     writer = tf.summary.FileWriter(os.path.join(classifier.model_dir, 'eval'), sess.graph)
# #     writer.add_summary(sess.run(pr), global_step=0)
# #     writer.close()
​
Collapse 