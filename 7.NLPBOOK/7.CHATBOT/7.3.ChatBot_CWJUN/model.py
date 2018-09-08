#-*- coding: utf-8 -*-
import tensorflow as tf
import sys

from configs import DEFINES

def Model(features, labels, mode, params):
	print(params['vocabularyLength'])
	# features = {"input": input, "output": output} features['input']) features['output'])
	s_embeddings = tf.eye(num_rows = params['vocabularyLength'], dtype = tf.float32)
	print(s_embeddings.shape)
	s_embeddings = tf.get_variable(name = 's_embeddings', initializer = s_embeddings,
                                           trainable = False)
	print(s_embeddings.shape)
	print(features['input'].shape)
	s_batch = tf.nn.embedding_lookup(params = s_embeddings, ids = features['input'])
	print(s_batch.shape)
#####################################################################################################
	t_embeddings = tf.eye(num_rows = params['vocabularyLength'])
	t_embeddings = tf.get_variable(name = 't_embeddings',
                                         initializer = t_embeddings,
                                         trainable = False)
	t_batch = tf.nn.embedding_lookup(params = t_embeddings, ids = features['output'])
######################################################################################################
	rnnCell = tf.nn.rnn_cell.BasicLSTMCell(params['hiddenSize'])
######################################################################################################
	#encoder_cell_list = [rnnCell for i in range(params['layerSize'])]
	#encoder_multi_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell_list)
	encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell=rnnCell,
                                                             inputs=s_batch,
                                                             dtype=tf.float32)
#######################################################################################################    
	#decoder_cell_list = [rnnCell for i in range(params['layerSize'])]
	#decoder_multi_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell_list)
	decoder_initial_state = encoder_final_state
	decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(cell=rnnCell,
                                                             inputs=t_batch,
                                                             initial_state=decoder_initial_state,
                                                             dtype=tf.float32)
	logits = tf.layers.dense(decoder_outputs, params['vocabularyLength'])
	# decoder_predict = tf.argmax(decoder_outputs, 2)
	#rnnCell = tf.nn.rnn_cell.BasicLSTMCell(params['hiddenSize'])
	# if mode == tf.estimator.ModeKeys.TRAIN:
	# rnnCell = tf.nn.rnn_cell.DropoutWrapper(rnnCell, output_keep_prob=DEFINES.dropoutWidth)
	# encCell = tf.nn.rnn_cell.MultiRNNCell([rnnCell for _ in range(params['layerSize'])])
	# decCell = tf.nn.rnn_cell.MultiRNNCell([rnnCell for _ in range(params['layerSize'])])	
	# outputs, encStates = tf.nn.dynamic_rnn(encCell, encoderEmbedded, dtype=tf.float32)
	# outputs, decStates = tf.nn.dynamic_rnn(decCell, decoderEmbedded, dtype=tf.float32, initial_state=encStates)	
########################################################################################################
	if mode == tf.estimator.ModeKeys.PREDICT:
		predictions = {
			'class_ids': predictedClasses[:, tf.newaxis], # Classes 처리 필요
			'probabilities': tf.nn.softmax(logits),	# 처리 필요
			'logits': logits,	# 처리 필요
		}
		return tf.estimator.EstimatorSpec(mode, predictions=predictions)

	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
																	labels=labels)
	loss = tf.reduce_mean(cross_entropy)
########################################################################################################	
	#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
	# accuracy = tf.metrics.accuracy(labels=labels,	# 작업 필요
	# 								predictions=decoder_outputs,
	# 								name='accOp')

	# metrics = {'accuracy': accuracy}
	# tf.summary.scalar('accuracy', accuracy[1])
	
	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
########################################################################################################	
	assert mode == tf.estimator.ModeKeys.TRAIN

	optimizer = tf.train.AdamOptimizer(learning_rate=DEFINES.learningRate)
	trainOp = optimizer.minimize(cross_entropy, global_step=tf.train.get_global_step())	
########################################################################################################
	return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=trainOp)
