# -*- coding: utf-8 -*-
import tensorflow as tf
import sys

from configs import DEFINES
import numpy as np


def layer_norm(inputs, eps=1e-6):
    feature_shape = inputs.get_shape()[-1:]

    mean, var = tf.nn.moments(inputs, [-1], keep_dims=True)

    beta = tf.Variable(tf.zeros(feature_shape), trainable=False)
    gamma = tf.Variable(tf.ones(feature_shape), trainable=False)

    return gamma * (inputs - mean) / ((var ** .5) + eps) + beta


def sublayer_connection(inputs, sublayer):
    return layer_norm(inputs + tf.keras.layers.Dropout(0.2)(sublayer()))


def feed_forward(inputs, num_units):
    outputs = tf.keras.layers.Dense(num_units[0], activation=tf.nn.relu)(inputs)

    return tf.keras.layers.Dense(num_units[1])(outputs)


def positional_encoding(dim, sentence_length, dtype=tf.float32):
    encoded_vec = np.array([pos/np.power(10000, 2*i/dim)
                            for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


def scaled_dot_product_attention(query, key, value, masked=False):
    key_seq_length = float(key.get_shape().as_list()[-2])
    key = tf.transpose(key, perm=[0, 2, 1])
    outputs = tf.matmul(query, key) / tf.sqrt(key_seq_length)

    if masked:
        diag_vals = tf.ones_like(outputs[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

        paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

    attention_map = tf.nn.softmax(outputs)

    return tf.matmul(attention_map, value)


def multi_head_attention(query, key, value, heads, masked=False):
    feature_dim = query.get_shape().as_list()[-1]

    query = tf.keras.layers.Dense(feature_dim, activation=tf.nn.relu)(query)
    key = tf.keras.layers.Dense(feature_dim, activation=tf.nn.relu)(key)
    value = tf.keras.layers.Dense(feature_dim, activation=tf.nn.relu)(value)

    query = tf.concat(tf.split(query, heads, axis=-1), axis=0)
    key = tf.concat(tf.split(key, heads, axis=-1), axis=0)
    value = tf.concat(tf.split(value, heads, axis=-1), axis=0)

    attention_map = scaled_dot_product_attention(query, key, value, masked)

    attn_outputs = tf.concat(tf.split(attention_map, heads, axis=0), axis=-1)

    return attn_outputs


def encoder_module(inputs, num_units, heads):
    self_attn = sublayer_connection(inputs,
                                    lambda: multi_head_attention(inputs,
                                                                 inputs,
                                                                 inputs,
                                                                 heads))
    outputs = sublayer_connection(self_attn,
                                  lambda: feed_forward(self_attn,
                                                       num_units))
    return outputs


def decoder_module(inputs, encoder_outputs, num_units, heads):
    masked_self_attn = sublayer_connection(inputs,
                                           lambda: multi_head_attention(inputs,
                                                                        inputs,
                                                                        inputs,
                                                                        heads,
                                                                        masked=True))
    self_attn = sublayer_connection(masked_self_attn,
                                    lambda: multi_head_attention(masked_self_attn,
                                                                 encoder_outputs,
                                                                 encoder_outputs,
                                                                 heads))
    outputs = sublayer_connection(self_attn,
                                  lambda: feed_forward(self_attn,
                                                       num_units))

    return outputs


def encoder(inputs, num_units, heads, num_layers):
    outputs = inputs
    for _ in range(num_layers):
        outputs = encoder_module(outputs, num_units, heads)

    return outputs


def decoder(inputs, encoder_outputs, num_units, heads, num_layers):
    outputs = inputs
    for _ in range(num_layers):
        outputs = decoder_module(outputs, encoder_outputs, num_units, heads)

    return outputs


def model_fn(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    positional_encoded = positional_encoding(params['embeddingSize'], DEFINES.maxSequenceLength)
    positional_encoded.trainable = False
    if TRAIN:
        position_inputs = tf.tile(tf.range(0, DEFINES.maxSequenceLength), [DEFINES.batchSize])
        position_inputs = tf.reshape(position_inputs, [DEFINES.batchSize, DEFINES.maxSequenceLength])
    else:
        position_inputs = tf.tile(tf.range(0, DEFINES.maxSequenceLength), [1])
        position_inputs = tf.reshape(position_inputs, [1, DEFINES.maxSequenceLength])

    embedding = tf.keras.layers.Embedding(params['vocabularyLength'],
                                          params['embeddingSize'])

    x_embedded_matrix = embedding(features['input'])
    y_embedded_matrix = embedding(features['output'])

    position_encode = tf.nn.embedding_lookup(positional_encoded, position_inputs)

    x_embedded_matrix = x_embedded_matrix + position_encode
    y_embedded_matrix = y_embedded_matrix + position_encode

    encoder_outputs = encoder(x_embedded_matrix,
                              [params['hiddenSize'] * 4, params['hiddenSize']], 4, 2)
    decoder_outputs = decoder(y_embedded_matrix,
                              encoder_outputs,
                              [params['hiddenSize'] * 4, params['hiddenSize']], 4, 2)

    logits = tf.keras.layers.Dense(params['vocabularyLength'])(decoder_outputs)

    predict = tf.argmax(logits, 2)

    if PREDICT:
        predictions = {
            'indexs': predict,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predict, name='accOp')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate=DEFINES.learningRate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)