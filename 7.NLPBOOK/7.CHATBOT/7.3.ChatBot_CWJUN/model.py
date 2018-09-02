#-*- coding: utf-8 -*-
import tensorflow as tf


from configs import DEFINES

def Model(features, labels, mode, params):
	# 임베딩 처리 
	#net = tf.feature_column.input_layer(features)
	print("+++++++++++++++++++++++++++++++++++++++++++++++++++")

	rnnCell = tf.nn.rnn_cell.BasicLSTMCell(params['hiddenSize'])
	
	if mode == tf.estimator.ModeKeys.TRAIN:
		rnnCell = tf.nn.rnn_cell.DropoutWrapper(rnnCell, output_keep_prob=DEFINES.dropoutWidth)

	encCell = tf.nn.rnn_cell.MultiRNNCell([rnnCell for _ in range(params['layerSize'])])		
	decCell = tf.nn.rnn_cell.MultiRNNCell([rnnCell for _ in range(params['layerSize'])])		
	outputs, encStates = tf.nn.dynamic_rnn(encCell, endInput, dtype=tf.float32)
	outputs, decStates = tf.nn.dynamic_rnn(decCell, decInput, dtype=tf.float32, initial_state=encStates)

########################################################################################################
# 정리 필요
	# Den 필요
	# 
	#timeStpes = tf.shape(outputs)[1]
	#outputs = tf.reshape(outputs, [-1, params['hiddenSize']])

	#logits = tf.matmul(outputs, self.weights) + self.bias
	#logits = tf.reshape(logits, [-1, timeStpes, self.vocab_size])

	predictedClasses = tf.argmax(logits, 2)
#########################################################################################################
	if mode == tf.estimator.ModeKeys.PREDICT:
		predictions = {
			'class_ids': predictedClasses[:, tf.newaxis],
			'probabilities': tf.nn.softmax(logits),
			'logits': logits,
		}
		return tf.estimator.EstimatorSpec(mode, predictions=predictions)

	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

	accuracy = tf.metrics.accuracy(labels=labels,
									predictions=predictedClasses,
									name='accOp')

	metrics = {'accuracy': accuracy}
	tf.summary.scalar('accuracy', accuracy[1])

	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
	
	assert mode == tf.estimator.ModeKeys.TRAIN

	optimizer = tf.train.AdamOptimizer(learning_rate=DEFINES.learningRate)
	trainOp = optimizer.minimize(loss, global_step=tf.train.get_global_step())
	
	return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=trainOp)
