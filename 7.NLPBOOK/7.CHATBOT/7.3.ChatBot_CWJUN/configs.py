#-*- coding: utf-8 -*-
import tensorflow as tf

tf.app.flags.DEFINE_integer('batchSize', 100, 'batch size')
tf.app.flags.DEFINE_integer('trainSteps', 10000, 'train steps')
tf.app.flags.DEFINE_float('dropoutWidth', 0.5, 'dropout width')
tf.app.flags.DEFINE_integer('layerSize', 5, 'layer size')
tf.app.flags.DEFINE_integer('hiddenSize', 128, 'weights size')
tf.app.flags.DEFINE_float('learningRate', 1e-3, 'learning rate')
tf.app.flags.DEFINE_string('dataPath', './data/ChatBotData.csv', 'data path')
tf.app.flags.DEFINE_string('vocabularyPath', './data/vocabularyData.voc', 'vocabulary path')
tf.app.flags.DEFINE_integer('shuffleSeek', 1000, 'shuffle random seek')
tf.app.flags.DEFINE_integer('maxSequenceLength', 15, 'max sequence length')

# Define FLAGS
DEFINES = tf.app.flags.FLAGS
