# -*- coding: utf-8 -*-

import tensorflow as tf
import random
import math
import os

from config import FLAGS
from model import Seq2Seq
from dialog import Dialog


def train(dialog, batch_size=100, epoch=100):
    model = Seq2Seq(dialog.vocab_size)

    with tf.Session() as sess:

        # 학습된 모델이 저장된 경로 체크
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("새로운 모델을 생성하는 중 입니다.")
            sess.run(tf.global_variables_initializer())

        # 로그를 저장
        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # 전체 batch size 결정
        total_batch = int(math.ceil(len(dialog.examples) / float(batch_size)))

        # total step
        print(total_batch * epoch)

        # 신경망 모델 학습
        for step in range(total_batch * epoch):
            enc_input, dec_input, targets = dialog.next_batch(batch_size)

            # model 학습
            _, loss = model.train(sess, enc_input, dec_input, targets)

            # log 출력
            if (step + 1) % 100 == 0:
                model.write_logs(sess, writer, enc_input, dec_input, targets)
                print('Step:', '%06d' % model.global_step.eval(), \
                      'cost =', '{:.6f}'.format(loss))

        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.ckpt_name)
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

    print('최적화 완료!')


def main(_):
    dialog = Dialog()

    dialog.load_vocab(FLAGS.voc_path)  # 어절 사전 파일 위치
    dialog.load_dialogue(FLAGS.data_path)  # 대화 스크립트 파일 위치
    train(dialog, batch_size=FLAGS.batch_size, epoch=FLAGS.epoch)  # 학습 시작


if __name__ == "__main__":
    tf.app.run()
