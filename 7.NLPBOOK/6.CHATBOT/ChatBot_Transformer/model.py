#-*- coding: utf-8 -*-
import tensorflow as tf
import sys
import numpy as np
from configs import DEFINES

# FFN 만드는 부분
def FFN(inputs):
    with tf.variable_scope('FFN_scope', reuse=tf.AUTO_REUSE):
        output = tf.layers.dense(inputs, DEFINES.embedding_size / 2, activation=tf.nn.relu)
        output =tf.layers.dense(output, DEFINES.embedding_size)
        return tf.keras.layers.Dropout(rate=DEFINES.dropout_width)(output)   

# 포지션 인코딩 하는 부분
def positional_encoding(dim, sentence_length, dtype=tf.float32):
    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)

# 인코더 모델 빌드
def encoder_model_build(encoder_inputs):
    o1 = tf.identity(encoder_inputs)

    for i in range(1, DEFINES.layers_size + 1):
        with tf.variable_scope(f"layer-{i}"):
            o2 = add_and_norm(o1, self_attention(q=o1,k=o1,v=o1), num=1)
            o3 = add_and_norm(o2, FFN(o2), num=2)
            o1 = tf.identity(o3)
    return o3

# self attention 동작
def self_attention(q, k, v):
    with tf.variable_scope("self-attention"):
        return multi_head(q, k, v, False)

# multi head 만드는 곳
def multi_head(q, k, v, mask):
    q, k, v = linear_projection(q, k, v)
    qs, ks, vs = split_heads(q, k, v)
    outputs = scaled_dot_product(qs, ks, vs, mask)
    output = concat_heads(outputs)
    output = tf.layers.dense(output, DEFINES.embedding_size)

    return tf.keras.layers.Dropout(rate=DEFINES.dropout_width)(output)

def linear_projection(q, k, v):
    q = tf.layers.dense(q, DEFINES.query_dimention, use_bias=False)
    k = tf.layers.dense(k, DEFINES.key_dimention, use_bias=False)
    v = tf.layers.dense(v, DEFINES.value_dimention, use_bias=False)
    return q, k, v

def split_heads(q, k, v):

    def split_last_dimension_then_transpose(tensor, num_heads, dim):
        t_shape = tensor.get_shape().as_list()
        tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, dim // num_heads])
        return tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, num_heads, max_seq_len, dim]

    qs = split_last_dimension_then_transpose(q, DEFINES.heads_size, DEFINES.query_dimention)
    ks = split_last_dimension_then_transpose(k, DEFINES.heads_size, DEFINES.key_dimention)
    vs = split_last_dimension_then_transpose(v, DEFINES.heads_size, DEFINES.value_dimention)

    return qs, ks, vs

def scaled_dot_product(qs, ks, vs, mask):
    key_dim_per_head = DEFINES.key_dimention // DEFINES.heads_size

    o1 = tf.matmul(qs, ks, transpose_b=True)
    o2 = o1 / (key_dim_per_head**0.5)

    if mask:
        diag_vals = tf.ones_like(o2[0, 0, :, :]) # (batch_size, num_heads, query_dim, key_dim)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (q_dim, k_dim)
        masks = tf.tile(tf.reshape(tril, [1, 1] + tril.get_shape().as_list()),
                        [tf.shape(o2)[0], tf.shape(o2)[1], 1, 1])
        paddings = tf.ones_like(masks) * -1e9
        o2 = tf.where(tf.equal(masks, 0), paddings, o2)

    o3 = tf.nn.softmax(o2)
    return tf.matmul(o3, vs)

def concat_heads(outputs):

    def transpose_then_concat_last_two_dimenstion(tensor):
        tensor = tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, max_seq_len, num_heads, dim]
        t_shape = tensor.get_shape().as_list()
        num_heads, dim = t_shape[-2:]
        return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

    return transpose_then_concat_last_two_dimenstion(outputs)

def add_and_norm( x, sub_layer_x, num=0):
    with tf.variable_scope(f"add-and-norm-{num}"):
        return tf.contrib.layers.layer_norm(tf.add(x, sub_layer_x)) # with Residual connection

# 디코드 모델 빌드
def decoder_model_build(decoder_inputs, encoder_outputs):
    o1 = tf.identity(decoder_inputs)

    for i in range(1, DEFINES.layers_size+1):
        with tf.variable_scope(f"layer-{i}"):
            o2 = add_and_norm(o1, masked_self_attention(q=o1,k=o1,v=o1), num=3)
            o3 = add_and_norm(o2, encoder_decoder_attention(q=o2,k=encoder_outputs,v=encoder_outputs), num=4)
            o4 = add_and_norm(o3, FFN(o3), num=5)
            o1 = tf.identity(o4)

    return o4

def masked_self_attention( q, k, v):
    with tf.variable_scope("masked-self-attention"):
        return multi_head(q, k, v, True)

def encoder_decoder_attention(q, k, v):
    with tf.variable_scope("encoder-decoder-attention"):
        return multi_head(q, k, v, False)

# def build_output(decoder_outputs, vocabulary_length, reuse=False):
#     with tf.variable_scope("Output", reuse=reuse):
#         logits = tf.layers.dense(decoder_outputs, vocabulary_length)

#     train_predictions = tf.argmax(logits[0], axis=1, name="train/pred_0")
#     return logits, train_predictions

# 에스티메이터 모델 부분이다.
# freatures : tf.data.Dataset.map을 통해서 만들어진 
# features = {"input": input, "output": output}
# labels : tf.data.Dataset.map을 통해서 만들어진 target
# mode는 에스티메이터 함수를 호출하면 에스티메이터 
# 프레임워크 모드의 값이 해당 부분이다.
# params : 에스티메이터를 구성할때 params 값들이다. 
# (params={ # 모델 쪽으로 파라메터 전달한다.)
def Model(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    initializer = tf.contrib.layers.xavier_initializer()
    embedding_encoder = tf.get_variable(name = "embedding_encoder", # 이름
                                    shape=[params['vocabulary_length'], params['embedding_size']], #  모양
                                    dtype=tf.float32, # 타입
                                    initializer=initializer, # 초기화 값
                                    trainable=True) # 학습 유무

    embedding_encoder = tf.nn.embedding_lookup(params = embedding_encoder, ids = features['input'])
    
    #position inputs 처리
    positional_encoded = positional_encoding(params['embedding_size'], params['max_sequence_length'])
    position_inputs = tf.tile(tf.range(0, params['max_sequence_length']), [DEFINES.batch_size])
    position_inputs = tf.reshape(position_inputs, [DEFINES.batch_size, params['max_sequence_length']])
    print("============================================")
    print(embedding_encoder)
    print(tf.nn.embedding_lookup(positional_encoded, position_inputs))
    print("============================================")
    encoded_inputs = tf.add(embedding_encoder,tf.nn.embedding_lookup(positional_encoded, position_inputs))
    encoder_outputs = tf.keras.layers.Dropout(rate=DEFINES.dropout_width)(encoded_inputs)   
    # 모델 만들기 완료    
    encoder_outputs = encoder_model_build(encoder_outputs)

    # 디코더 부분
    initializer = tf.contrib.layers.xavier_initializer()
    embedding_decoder = tf.get_variable(name = "embedding_decoder", # 이름
                                    shape=[params['vocabulary_length'], params['embedding_size']], # 모양
                                    dtype=tf.float32, # 타입
                                    initializer=initializer, # 초기화 값
                                    trainable=True)  # 학습 유무 
    embedding_decoder = tf.nn.embedding_lookup(params = embedding_decoder, ids = features['output'])

    #position inputs 처리
    positional_encoded = positional_encoding(params['embedding_size'], params['max_sequence_length'])
    position_inputs = tf.tile(tf.range(0, params['max_sequence_length']), [DEFINES.batch_size])
    position_inputs = tf.reshape(position_inputs, [DEFINES.batch_size, params['max_sequence_length']])

    decoder_inputs = tf.add(embedding_decoder,tf.nn.embedding_lookup(positional_encoded, position_inputs))
    decoder_outputs = tf.keras.layers.Dropout(rate=DEFINES.dropout_width)(encoded_inputs)

    # 모델 만들기 완료    
    decoder_outputs = decoder_model_build(decoder_outputs, encoder_outputs)

    logits = tf.layers.dense(decoder_outputs, params['vocabulary_length'], activation=None)
    #logits, _ = build_output(decoder_outputs, params['vocabulary_length']) # logits, train_predictions
	# 예측한 값을 argmax를 통해서 가져 온다.
    predict = tf.argmax(logits, 2)
    #predict = tf.argmax(logits, axis=2)
    # 예측 mode 확인 부분이며 예측은 여기 까지 수행하고 리턴한다.
    if PREDICT:
        predictions = { # 예측 값들이 여기에 딕셔너리 형태로 담긴다.
            'indexs': predict, # 시퀀스 마다 예측한 값
            'logits': logits, # 마지막 결과 값
        }
        # 에스티메이터에서 리턴하는 값은 tf.estimator.EstimatorSpec 
        # 객체를 리턴 한다.
        # mode : 에스티메이터가 수행하는 mode (tf.estimator.ModeKeys.PREDICT)
        # predictions : 예측 값
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    # 마지막 결과 값과 정답 값을 비교하는 
    # tf.nn.sparse_softmax_cross_entropy_with_logits(로스함수)를 
    # 통과 시켜 틀린 만큼의
    # 에러 값을 가져 오고 이것들은 차원 축소를 통해 단일 텐서 값을 반환 한다.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    # 라벨과 결과가 일치하는지 빈도 계산을 통해 
    # 정확도를 측정하는 방법이다.
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predict,name='accOp')

    # 정확도를 전체값으로 나눈 값이다.
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    
    # 평가 mode 확인 부분이며 평가는 여기 까지 
    # 수행하고 리턴한다.
    if EVAL:
        # 에스티메이터에서 리턴하는 값은 
        # tf.estimator.EstimatorSpec 객체를 리턴 한다.
        # mode : 에스티메이터가 수행하는 mode (tf.estimator.ModeKeys.EVAL)
        # loss : 에러 값
        # eval_metric_ops : 정확도 값
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # 파이썬 assert구문으로 거짓일 경우 프로그램이 종료 된다.
    # 수행 mode(tf.estimator.ModeKeys.TRAIN)가 
    # 아닌 경우는 여기 까지 오면 안되도록 방어적 코드를 넣은것이다.
    assert TRAIN

    # 아담 옵티마이저를 사용한다.
    optimizer = tf.train.AdamOptimizer(learning_rate=DEFINES.learning_rate)
    # 에러값을 옵티마이저를 사용해서 최소화 시킨다.
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())  
    # 에스티메이터에서 리턴하는 값은 tf.estimator.EstimatorSpec 객체를 리턴 한다.
    # mode : 에스티메이터가 수행하는 mode (tf.estimator.ModeKeys.EVAL)
    # loss : 에러 값
    # train_op : 그라디언트 반환
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
