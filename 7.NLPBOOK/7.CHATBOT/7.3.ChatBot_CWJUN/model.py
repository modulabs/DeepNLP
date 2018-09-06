#-*- coding: utf-8 -*-
import tensorflow as tf
import sys

from configs import DEFINES

def Model(features, labels, mode, params):
	print("++++++++++++++++++++++++++++++++++++++++++++++++++11")
	print(params['vocabularyLength'])
	# features = {"input": input, "output": output} features['input']) features['output'])
	s_embeddings = tf.eye(num_rows = params['vocabularyLength'], dtype = tf.float32)
	print(s_embeddings.shape)
	print("++++++++++++++++++++++++++++++++++++++++++++++++++21")
	s_embeddings = tf.get_variable(name = 's_embeddings', initializer = s_embeddings,
                                           trainable = False)
	print(s_embeddings.shape)
	print(features['input'].shape)
	print("++++++++++++++++++++++++++++++++++++++++++++++++++31")
	s_batch = tf.nn.embedding_lookup(params = s_embeddings, ids = features['input'])
	print(s_batch.shape)
	print("++++++++++++++++++++++++++++++++++++++++++++++++++41")
#####################################################################################################
	t_embeddings = tf.eye(num_rows = params['vocabularyLength'])
	print("++++++++++++++++++++++++++++++++++++++++++++++++++51")
	t_embeddings = tf.get_variable(name = 't_embeddings',
                                         initializer = t_embeddings,
                                         trainable = False)
	print("++++++++++++++++++++++++++++++++++++++++++++++++++61")
	t_batch = tf.nn.embedding_lookup(params = t_embeddings, ids = features['output'])
	print("++++++++++++++++++++++++++++++++++++++++++++++++++71")
######################################################################################################
	rnnCell = tf.nn.rnn_cell.BasicLSTMCell(params['hiddenSize'])
	print("++++++++++++++++++++++++++++++++++++++++++++++++++81")
######################################################################################################
	#encoder_cell_list = [rnnCell for i in range(params['layerSize'])]
	print("++++++++++++++++++++++++++++++++++++++++++++++++++91")
	#encoder_multi_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell_list)
	print("++++++++++++++++++++++++++++++++++++++++++++++++++92")
	encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell=rnnCell,
                                                             inputs=s_batch,
                                                             dtype=tf.float32)
	print("++++++++++++++++++++++++++++++++++++++++++++++++++93")
#######################################################################################################    
	#decoder_cell_list = [rnnCell for i in range(params['layerSize'])]
	print("++++++++++++++++++++++++++++++++++++++++++++++++++94")
	#decoder_multi_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell_list)
	print("++++++++++++++++++++++++++++++++++++++++++++++++++95")
	decoder_initial_state = encoder_final_state
	print("++++++++++++++++++++++++++++++++++++++++++++++++++96")
	decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(cell=rnnCell,
                                                             inputs=t_batch,
                                                             initial_state=decoder_initial_state,
                                                             dtype=tf.float32)

	decoder_predict = tf.argmax(decoder_outputs, 2)
	print("++++++++++++++++++++++++++++++++++++++++++++++++++97")

	#rnnCell = tf.nn.rnn_cell.BasicLSTMCell(params['hiddenSize'])
	
	# if mode == tf.estimator.ModeKeys.TRAIN:
	# 	rnnCell = tf.nn.rnn_cell.DropoutWrapper(rnnCell, output_keep_prob=DEFINES.dropoutWidth)

	# encCell = tf.nn.rnn_cell.MultiRNNCell([rnnCell for _ in range(params['layerSize'])])
	# decCell = tf.nn.rnn_cell.MultiRNNCell([rnnCell for _ in range(params['layerSize'])])	
	# outputs, encStates = tf.nn.dynamic_rnn(encCell, encoderEmbedded, dtype=tf.float32)
	# outputs, decStates = tf.nn.dynamic_rnn(decCell, decoderEmbedded, dtype=tf.float32, initial_state=encStates)
	print("+++++++++++++++++++++++++++++++++++++++++++++++++++12")
	
########################################################################################################
# 정리 필요
	# Den 필요

	#predictedClasses = tf.argmax(logits, 2)
#########################################################################################################
	print("+++++++++++++++++++++++++++++++++++++++++++++++++++13")
	if mode == tf.estimator.ModeKeys.PREDICT:
		predictions = {
			'class_ids': predictedClasses[:, tf.newaxis],
			'probabilities': tf.nn.softmax(logits),
			'logits': logits,
		}
		return tf.estimator.EstimatorSpec(mode, predictions=predictions)
	print("+++++++++++++++++++++++++++++++++++++++++++++++++++14")
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decoder_outputs,
																	labels=labels)
	loss = tf.reduce_mean(cross_entropy)
	#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
	print("+++++++++++++++++++++++++++++++++++++++++++++++++++15")
	# accuracy = tf.metrics.accuracy(labels=labels,
	# 								predictions=decoder_outputs,
	# 								name='accOp')

	# metrics = {'accuracy': accuracy}
	# tf.summary.scalar('accuracy', accuracy[1])
	
	print("+++++++++++++++++++++++++++++++++++++++++++++++++++16")
	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
	
	assert mode == tf.estimator.ModeKeys.TRAIN

	optimizer = tf.train.AdamOptimizer(learning_rate=DEFINES.learningRate)
	trainOp = optimizer.minimize(cross_entropy, global_step=tf.train.get_global_step())	
	print("+++++++++++++++++++++++++++++++++++++++++++++++++++17")
	return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=trainOp)
