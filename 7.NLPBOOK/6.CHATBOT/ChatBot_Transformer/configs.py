#-*- coding: utf-8 -*-
import tensorflow as tf

tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size') # 배치 크기
tf.app.flags.DEFINE_integer('train_steps', 1, 'train steps') # 학습 에포크
tf.app.flags.DEFINE_float('dropout_width', 0.5, 'dropout width') # 드롭아웃 크기
tf.app.flags.DEFINE_integer('hidden_size', 128, 'weights size') # 가중치 크기
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate') # 학습률
tf.app.flags.DEFINE_integer('shuffle_seek', 1000, 'shuffle random seek') # 셔플 시드값
tf.app.flags.DEFINE_integer('max_sequence_length', 25, 'max sequence length') # 시퀀스 길이
tf.app.flags.DEFINE_integer('embedding_size', 128, 'embedding size') # 임베딩 크기
tf.app.flags.DEFINE_integer('query_dimention', 128, 'query dimention')
tf.app.flags.DEFINE_integer('key_dimention', 128, 'key dimention')
tf.app.flags.DEFINE_integer('value_dimention', 128, 'value dimention')
tf.app.flags.DEFINE_integer('layers_size', 8, 'layers size')
tf.app.flags.DEFINE_integer('heads_size', 8, 'heads size')
tf.app.flags.DEFINE_string('data_path', './data_in/ChatBotData.csv', 'data path') #  데이터 위치
tf.app.flags.DEFINE_string('vocabulary_path', './data_out/vocabularyData.voc', 'vocabulary path') # 사전 위치
tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point', 'check point path') # 체크 포인트 위치
tf.app.flags.DEFINE_boolean('tokenize_as_morph', False, 'set morph tokenize') # 형태소에 따른 토크나이징 사용 유무
tf.app.flags.DEFINE_boolean('masks', False, 'set masks')


# Define FLAGS
DEFINES = tf.app.flags.FLAGS
