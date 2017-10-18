# -*- coding: utf-8 -*-
# 자세한 설명은 상위 폴더의 03 - Seq2Seq.py 등에서 찾으실 수 있습니다.

import tensorflow as tf


# Seq2Seq 기본 클래스
class Seq2Seq:
    logits = None
    outputs = None
    cost = None
    train_op = None

    def __init__(self, vocab_size, n_hidden=128, n_layers=3):
        self.learning_late = 0.001

        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.enc_input = tf.placeholder(tf.float32, [None, None, self.vocab_size])
        self.dec_input = tf.placeholder(tf.float32, [None, None, self.vocab_size])
        self.targets = tf.placeholder(tf.int64, [None, None])

        self.weights = tf.Variable(tf.ones([self.n_hidden, self.vocab_size]), name="weights")
        self.bias = tf.Variable(tf.zeros([self.vocab_size]), name="bias")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        self.build_model()

        self.saver = tf.train.Saver(tf.global_variables())

    # model 생성
    def build_model(self):
        # enc_input, dec_input 대해서 transpose를 수행
        self.enc_input = tf.transpose(self.enc_input, [1, 0, 2])
        self.dec_input = tf.transpose(self.dec_input, [1, 0, 2])

        # Cell 생성 후 저장
        enc_cell, dec_cell = self.build_cells()

        # 모든 Tensorflow의 RNN 함수들은 Cell을 인자로 받음
        with tf.variable_scope('encode'):
            outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, self.enc_input, dtype=tf.float32)
        with tf.variable_scope('decode'):
            outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, self.dec_input, dtype=tf.float32)

        self.logits, self.cost, self.train_op = self.build_ops(outputs, self.targets)

        self.outputs = tf.argmax(self.logits, 2)

    # cell 생성
    def build_cells(self, output_keep_prob=0.5):
        # tensorflow 1.0.1버전 수정 (오연택)

        # LSTM cell
        enc_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden);

        # RNN cell의 입력과 출력 연결에 대해 dropout 기능을 추가해주는 wrapper
        # 다중 레이어와 과적합 방지를 위한 Dropout 기법을 사용
        enc_cell = tf.contrib.rnn.DropoutWrapper(enc_cell, output_keep_prob=output_keep_prob)

        # 여러개의 RNN cell을 연결하여, Multi-layer cell로 구성해주는 wrapper입니다.
        enc_cell = tf.contrib.rnn.MultiRNNCell([enc_cell] * self.n_layers)

        # decoding cell
        dec_cell = tf.contrib.rnn.BasicRNNCell(self.n_hidden)
        dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=output_keep_prob)
        dec_cell = tf.contrib.rnn.MultiRNNCell([dec_cell] * self.n_layers)

        # 생성된 cell retrun
        return enc_cell, dec_cell

    # logits, loss, optimizer 정의
    def build_ops(self, outputs, targets):
        time_steps = tf.shape(outputs)[1]
        outputs = tf.reshape(outputs, [-1, self.n_hidden])

        # logits 는 one-hot 인코딩을 사용합니다.
        logits = tf.matmul(outputs, self.weights) + self.bias
        logits = tf.reshape(logits, [-1, time_steps, self.vocab_size])

        ## Define loss and optimizer
        # Tensorflow 1.0.0에서는 파라미터 순서가 변경
        cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                           labels=targets))  # Softmax loss
        # AdamOptimizer
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_late).minimize(cost, global_step=self.global_step)

        tf.summary.scalar('cost', cost)

        return logits, cost, train_op

    # train 함수
    def train(self, session, enc_input, dec_input, targets):
        return session.run([self.train_op, self.cost],
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input,
                                      self.targets: targets})

    def predict(self, session, enc_input, dec_input):
        return session.run(self.outputs,
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input})

    def write_logs(self, session, writer, enc_input, dec_input, targets):
        merged = tf.summary.merge_all()

        summary = session.run(merged, feed_dict={self.enc_input: enc_input,
                                                 self.dec_input: dec_input,
                                                 self.targets: targets})

        writer.add_summary(summary, self.global_step.eval())
