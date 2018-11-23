# -*- coding: utf-8 -*-
import tensorflow as tf
import sys

from configs import DEFINES
import numpy as np


def layer_norm(inputs, eps=1e-6):
    # LayerNorm(x + Sublayer(x))
    feature_shape = inputs.get_shape()[-1:]
    #  평균과 표준편차을 넘겨 준다.
    mean = tf.keras.backend.mean(inputs, [-1], keepdims=True)
    std = tf.keras.backend.std(inputs, [-1], keepdims=True)
    beta = tf.Variable(tf.zeros(feature_shape), trainable=False)
    gamma = tf.Variable(tf.ones(feature_shape), trainable=False)

    return gamma * (inputs - mean) / (std + eps) + beta


def sublayer_connection(inputs, sublayer):
    # LayerNorm(x + Sublayer(x))
    return tf.keras.layers.Dropout(rate=DEFINES.dropout_width)(layer_norm(inputs + sublayer)) 


def feed_forward(inputs, num_units):
    # FFN(x) = max(0, xW1 + b1)W2 + b2 
    with tf.variable_scope("feed_forward", reuse=tf.AUTO_REUSE):
        outputs = tf.keras.layers.Dense(num_units[0], activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dropout(rate=DEFINES.dropout_width)(outputs)
        return tf.keras.layers.Dense(num_units[1])(outputs)

def conv_1d_layer(inputs, num_units):
    # Another way of describing this is as two convolutions with kernel size 1
    with tf.variable_scope("conv_1d_layer", reuse=tf.AUTO_REUSE):
        outputs = tf.keras.layers.Conv1D(num_units[0], kernel_size = 1, activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dropout(rate=DEFINES.dropout_width)(outputs)
        return tf.keras.layers.Conv1D(num_units[1], kernel_size = 1)(outputs)


def positional_encoding(dim, sentence_length, dtype=tf.float32):
    #Positional Encoding
    # paper: https://arxiv.org/abs/1706.03762
    # P E(pos,2i) = sin(pos/100002i/dmodel)
    # P E(pos,2i+1) = cos(pos/100002i/dmodel)
    encoded_vec = np.array([pos/np.power(10000, 2*i/dim)
                            for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


def scaled_dot_product_attention(query, key, value, masked=False):
    #Attention(Q, K, V ) = softmax(QKt / root dk)V
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
    # MultiHead(Q, K, V ) = Concat(head1, ..., headh)WO
    with tf.variable_scope("multi_head_attention", reuse=tf.AUTO_REUSE):
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
        multi_head_attention(inputs, inputs, inputs, heads))
    
    if DEFINES.conv_1d_layer:
        network_layer = conv_1d_layer(self_attn, num_units)
    else:
        network_layer = feed_forward(self_attn, num_units)
        
    
    outputs = sublayer_connection(self_attn, network_layer)
    return outputs


def decoder_module(inputs, encoder_outputs, num_units, heads):
    # sublayer_connection Parameter input Self-Attention
    # multi_head_attention parameter Query Key Value Head masked
    masked_self_attn = sublayer_connection(inputs, 
        multi_head_attention(inputs, inputs, inputs, heads, masked=True))
    
    self_attn = sublayer_connection(masked_self_attn, 
        multi_head_attention(masked_self_attn, encoder_outputs, encoder_outputs, heads))
    
    if DEFINES.conv_1d_layer:
        network_layer = conv_1d_layer(self_attn, num_units)        
    else:
        network_layer = feed_forward(self_attn, num_units)
    
    outputs = sublayer_connection(self_attn, network_layer)
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


def Model(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    positional_encoded = positional_encoding(params['embedding_size'], DEFINES.max_sequence_length)
    positional_encoded.trainable = False
    if TRAIN:
        position_inputs = tf.tile(tf.range(0, DEFINES.max_sequence_length), [DEFINES.batch_size])
        position_inputs = tf.reshape(position_inputs, [DEFINES.batch_size, DEFINES.max_sequence_length])
    else:
        position_inputs = tf.tile(tf.range(0, DEFINES.max_sequence_length), [1])
        position_inputs = tf.reshape(position_inputs, [1, DEFINES.max_sequence_length])

    if DEFINES.xavier_embedding:
        embedding = tf.get_variable(name ='embedding', dtype=tf.float32,
                                            shape=[params['vocabulary_length'], params['embedding_size']],
                                            initializer = tf.contrib.layers.xavier_initializer())
        encoder_inputs = tf.nn.embedding_lookup(ids = features['input'], params = embedding)
        decoder_inputs = tf.nn.embedding_lookup(ids = features['output'], params = embedding)
    else:
        embedding = tf.keras.layers.Embedding(params['vocabulary_length'],params['embedding_size'])
        encoder_inputs = embedding(features['input'])
        decoder_inputs = embedding(features['output'])

    position_encode = tf.nn.embedding_lookup(positional_encoded, position_inputs)

    encoder_inputs = encoder_inputs + position_encode
    decoder_inputs = decoder_inputs + position_encode
    # dmodel = 512, inner-layer has dimensionality df f = 2048.  (512 * 4)
    # dmodel = 128 , inner-layer has dimensionality df f = 512  (128 * 4)
    # H = 8 N = 6
    # H = 4 N = 2
    encoder_outputs = encoder(encoder_inputs,
                              [params['hidden_size'] *  4, params['hidden_size']], DEFINES.heads_size, DEFINES.layers_size)
    decoder_outputs = decoder(decoder_inputs,
                              encoder_outputs,
                              [params['hidden_size'] *  4, params['hidden_size']], DEFINES.heads_size, DEFINES.layers_size)

    logits = tf.keras.layers.Dense(params['vocabulary_length'])(decoder_outputs)

    predict = tf.argmax(logits, 2)

    if PREDICT:
        predictions = {
            'indexs': predict,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # if DEFINES.mask_loss:
    #     embedding_tile = tf.tile(tf.expand_dims(embedding, 0), [DEFINES.batch_size, 1, 1])
    #     linear_outputs = tf.matmul(decoder_outputs, embedding_tile, transpose_b = True) 

    #     #mask_zero = 1 - tf.cast(tf.equal(labels, 0),dtype=tf.float32)
    #     mask_end = 1 - tf.cast(tf.equal(labels, 2), dtype=tf.float32)
    #     labels_one_hot = tf.one_hot(indices = labels, depth = params['vocabulary_length'], dtype = tf.float32) # [BS, senxlen, vocab_size]
    #     loss = tf.nn.softmax_cross_entropy_with_logits(labels = labels_one_hot, logits = linear_outputs)
    #     #loss = loss * mask_zero
    #     loss = loss * mask_end
    #     loss = tf.reduce_mean(loss)
    # else:
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predict, name='accOp')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert TRAIN

    # lrate = d−0.5 *  model · min(step_num−0.5, step_num · warmup_steps−1.5)
    optimizer = tf.train.AdamOptimizer(learning_rate=DEFINES.learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
