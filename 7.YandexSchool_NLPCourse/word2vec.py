import tensorflow as tf
import numpy as np

class Word2Vec(tf.estimator.Estimator):
    """
    Wrod Embedding 방법 중 기본이 되는 Word2Vec을 확인하기 위한 Class
    """
    def __init__(self, n_input, n_embed, n_sampl, learning_rate=0.1):
        """
        설명: 초기 생성 함수
        Parameters:
            - n_input: 입력 Vocabulary size
            - n_embed: embedding 사이즈 (hidden size)
            - n_sampl: sample size ???
        """
        self.n_input = n_input
        self.n_embed = n_embed
        self.n_sampl = min(n_embed, n_sampl)
        self.learning_rate = learning_rate

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self._build_network()

    def _build_network(self):
        """
        설명: Tensorflow Graph 생성
        """
        self.inputs = tf.placeholder(tf.int32, shape=[None])
        self.labels = tf.placeholder(tf.int32, shape=[None, 1])

        self.embeddings = tf.Variable(tf.random_uniform([self.n_input, self.n_embed], -1.0, 1.0))

        self.selected_embed = tf.nn.embedding_lookup(self.embeddings, self.inputs)

        nce_weights = tf.Variable(tf.random_uniform([self.n_input, self.n_embed], -1.0, 1.0))
        nce_biases = tf.Variable(tf.zeros([self.n_input]))

        self.loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, self.labels, self.selected_embed, self.n_sampl, self.n_input))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

    def train(self, sess, inputs, labels):
        """
        설명: 학습
        Parameters:
            - sess: Tensorflow Session
            - inputs: 입력값
            - labels: 출력값
        Return:
            - otimizer: 학습결과
            - loss: 오차
        """
        return sess.run([self.optimizer, self.loss], feed_dict={self.inputs: inputs, self.labels: labels})

    def select(self, sess, inputs):
        """
        설명: 단어 선택
        Parameters:
            - sess: Tensorflow Session
            - inputs: 입력값
        Return: embedding 백터 값
        """
        return sess.run(self.selected_embed, feed_dict={self.inputs: inputs})

