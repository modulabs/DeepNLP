import tensorflow as tf
import model as ml
import data
import numpy as np
import os
import sys

from configs import DEFINES


def serving_input_receiver_fn():
    receiver_tensor = {
        'word': tf.placeholder(dtype=tf.int32, shape=[None, get_vocab_size()])
    }
    features = {
        key: tensor for key, tensor in receiver_tensor.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)


def main(self):
    # 데이터를 통한 사전 구성 한다.
    char2idx, idx2char, vocabularyLength = data.loadVocabulary()
    # 훈련 데이터와 테스트 데이터를 가져온다.
    xTrain, yTrain, xTest, yTest = data.loadData()

    # 훈련셋 인코딩 만드는 부분이다.
    inputTrainEnc, inputTrainEncLength = data.encProcessing(xTrain, char2idx)
    # 훈련셋 디코딩 입력 부분 만드는 부분이다.
    outputTrainDec, outputTrainDecLength = data.decOutputProcessing(yTrain, char2idx)
    # 훈련셋 디코딩 출력 부분 만드는 부분이다.
    targetTrainDec = data.decTargetProcessing(yTrain, char2idx)

    print(inputTrainEnc.shape, outputTrainDec.shape, targetTrainDec.shape)

    inputTestEnc, inputTestEncLength = data.encProcessing(xTest, char2idx)
    outputTestDec, outputTestDecLength = data.decOutputProcessing(yTest, char2idx)
    targetTestDec = data.decTargetProcessing(yTest, char2idx)

    # 현재 경로'./'에 현재 경로 하부에 
    # 체크 포인트를 저장한 디렉토리를 설정한다.
    checkPointPath = os.path.join(os.getcwd(), DEFINES.checkPointPath)
    # 디렉토리를 만드는 함수이며 두번째 인자 exist_ok가 
    # True이면 디렉토리가 이미 존재해도 OSError가 
    # 발생하지 않는다.
    # exist_ok가 False이면 이미 존재하면 
    # OSError가 발생한다.
    os.makedirs(checkPointPath, exist_ok=True)

    # 에스티메이터 구성한다.
    classifier = tf.estimator.Estimator(
        model_fn=ml.model_fn,  # 모델 등록한다.
        model_dir=DEFINES.checkPointPath,  # 체크포인트 위치 등록한다.
        params={  # 모델 쪽으로 파라메터 전달한다.
            'hiddenSize': DEFINES.hiddenSize,  # 가중치 크기 설정한다.
            'layerSize': DEFINES.layerSize,  # 멀티 레이어 층 개수를 설정한다.
            'learningRate': DEFINES.learningRate,  # 학습율 설정한다.
            'vocabularyLength': vocabularyLength,  # 딕셔너리 크기를 설정한다.
            'embeddingSize': DEFINES.embeddingSize,  # 임베딩 크기를 설정한다.
            'embedding': DEFINES.embedding,  # 임베딩 사용 유무를 설정한다.
            'multilayer': DEFINES.multilayer,  # 멀티 레이어 사용 유무를 설정한다.
            'max_sequence_length': DEFINES.maxSequenceLength
        })

    # 학습 실행
    print(inputTrainEnc.shape, outputTrainDec.shape, targetTrainDec.shape)
    classifier.train(input_fn=lambda: data.trainInputFn(
        inputTrainEnc, outputTrainDec, targetTrainDec, DEFINES.batchSize), steps=DEFINES.trainSteps)

    evalResult = classifier.evaluate(input_fn=lambda: data.evalInputFn(
        inputTrainEnc, outputTrainDec, targetTrainDec, 1))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**evalResult))

    # outputPredicDec, outputPredicDecLength = data.decOutputProcessing([""] * 100, char2idx)
    # trainResult = classifier.evaluate(input_fn=lambda: data.trainInputFn(
    #     inputTrainEnc, outputPredicDec, targetTrainDec, DEFINES.batchSize))
    # print('\nTrain set accuracy: {accuracy:0.3f}\n'.format(**trainResult))

    # Save 
    # save_model_path = classifier.export_savedmodel(
    #     export_dir_base=DEFINES.saveModelPath,
    #     serving_input_receiver_fn=serving_input_receiver_fn)

    # 테스트용 데이터 만드는 부분이다.
    # 인코딩 부분 만든다.
    inputPredicEnc, inputPredicEncLength = data.encProcessing(["가끔 궁금해"], char2idx)
    # 학습 과정이 아니므로 디코딩 입력은 
    # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
    outputPredicDec, outputPredicDecLength = data.decOutputProcessing([""], char2idx)
    # 학습 과정이 아니므로 디코딩 출력 부분도 
    # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
    targetPredicDec = data.decTargetProcessing([""], char2idx)

    for i in range(DEFINES.maxSequenceLength):
        if i > 0:
            outputPredicDec, outputPredicDecLength = data.decOutputProcessing([answer], char2idx)
            targetPredicDec = data.decTargetProcessing([answer], char2idx)
        # 예측을 하는 부분이다.
        predictions = classifier.predict(
            input_fn=lambda: data.evalInputFn(inputPredicEnc, outputPredicDec, targetPredicDec, DEFINES.batchSize))

        # 예측한 값을 인지 할 수 있도록
        # 텍스트로 변경하는 부분이다.
        answer = data.pred2string(predictions, idx2char)

    print("final answer:", anwer)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

tf.logging.set_verbosity
