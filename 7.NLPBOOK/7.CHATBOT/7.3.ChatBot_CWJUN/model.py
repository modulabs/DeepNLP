#-*- coding: utf-8 -*-
import tensorflow as tf
import sys

from configs import DEFINES

def makeLSTMCell(hiddenSize):
    cell = tf.contrib.rnn.BasicLSTMCell(hiddenSize)
    return cell

def Model(features, labels, mode, params):
	print(params['vocabularyLength'])
	if params['embedding'] == True:
		initializer = tf.contrib.layers.xavier_initializer()
		embeddingEncoder = tf.get_variable(name = "embeddingEncoder",
									   shape=[params['vocabularyLength'], params['embeddingSize']],
									   dtype=tf.float32,
									   initializer=initializer,
									   trainable=True)
	else:	
		embeddingEncoder = tf.eye(num_rows = params['vocabularyLength'], dtype = tf.float32)
		embeddingEncoder = tf.get_variable(name = 'embeddingEncoder', initializer = embeddingEncoder,
                                           trainable = False)

	embeddingEncoderBatch = tf.nn.embedding_lookup(params = embeddingEncoder, ids = features['input'])
	print(embeddingEncoderBatch.shape)
#####################################################################################################
	if params['embedding'] == True:
		initializer = tf.contrib.layers.xavier_initializer()
		embeddingDecoder = tf.get_variable(name = "embeddingDecoder",
									   shape=[params['vocabularyLength'], params['embeddingSize']],
									   dtype=tf.float32,
									   initializer=initializer,
									   trainable=True)		
		embeddingDecoder = tf.get_variable(name = 'embeddingDecoder', initializer = embeddingDecoder,
                                           trainable = False)
	else:
		embeddingDecoder = tf.eye(num_rows = params['vocabularyLength'])
		embeddingDecoder = tf.get_variable(name = 'embeddingDecoder',
											initializer = embeddingDecoder,
											trainable = False)

	embeddingDecoderBatch = tf.nn.embedding_lookup(params = embeddingDecoder, ids = features['output'])
######################################################################################################
	with tf.variable_scope('encoderScope', reuse=tf.AUTO_REUSE):
		if params['multilayer'] == True:
			encoderCellList = [makeLSTMCell(params['hiddenSize']) for i in range(params['layerSize'])]
			rnnCell = tf.contrib.rnn.MultiRNNCell(encoderCellList)
		else:
			rnnCell = makeLSTMCell(params['hiddenSize'])

		encoderOutputs, encoderFinalState = tf.nn.dynamic_rnn(cell=rnnCell,
																inputs=embeddingEncoderBatch,
																dtype=tf.float32)
#######################################################################################################    
	with tf.variable_scope('decoderScope', reuse=tf.AUTO_REUSE):
		if params['multilayer'] == True:
			decoderCellList = [makeLSTMCell(params['hiddenSize']) for i in range(params['layerSize'])]
			rnnCell = tf.contrib.rnn.MultiRNNCell(decoderCellList)
		else:
			rnnCell = makeLSTMCell(params['hiddenSize'])

		decoderInitialState = encoderFinalState

		if params['attention'] == True:
			attention = tf.contrib.seq2seq.LuongAttention(num_units=params['hiddenSize'], 
															memory=encoderOutputs)
			# Attention 구성 중	


		decoderOutputs, decoderFinalState = tf.nn.dynamic_rnn(cell=rnnCell,
																inputs=embeddingDecoderBatch,
																initial_state=decoderInitialState,
																dtype=tf.float32)
#######################################################################################################
	logits = tf.layers.dense(decoderOutputs, params['vocabularyLength'], activation=None)

	predict = tf.argmax(logits, 2)
	#print("predict.shape")
	#print(predict.shape)
########################################################################################################
	if mode == tf.estimator.ModeKeys.PREDICT:
		predictions = {
			'class_ids': predict[:, params['maxSequenceLength']],
			'probabilities': tf.nn.softmax(logits),
			'logits': logits,
		}
		return tf.estimator.EstimatorSpec(mode, predictions=predictions)
########################################################################################################	
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
	accuracy = tf.metrics.accuracy(labels=labels, predictions=predict,name='accOp')

	metrics = {'accuracy': accuracy}
	tf.summary.scalar('accuracy', accuracy[1])
	
	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
########################################################################################################	
	assert mode == tf.estimator.ModeKeys.TRAIN

	optimizer = tf.train.AdamOptimizer(learning_rate=DEFINES.learningRate)
	trainOp = optimizer.minimize(loss, global_step=tf.train.get_global_step())	
########################################################################################################
	return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=trainOp)
