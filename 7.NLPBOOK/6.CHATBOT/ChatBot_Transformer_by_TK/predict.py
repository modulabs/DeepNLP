import tensorflow as tf
import data
import sys
import model as ml

from configs import DEFINES

classifier = None
char2idx, idx2char = None, None
vocabularyLength = None


PAD = "<PADDING>"
STD = "<START>"
END = "<END>"
UNK = "<UNKNWON>"

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

def run():
    sys.stdout.write("> ")
    sys.stdout.flush()
    line = sys.stdin.readline()

    while line:
        if line.strip() == 'quit':
            return
        pred(line.strip())

        sys.stdout.write("\n> ")
        sys.stdout.flush()

        line = sys.stdin.readline()


def pred(input):
    inputPredicEnc, inputPredicEncLength = data.encProcessing([input], char2idx, pred=True)
    print(inputPredicEnc)
    # 학습 과정이 아니므로 디코딩 입력은
    # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
    outputPredicDec, outputPredicDecLength = data.decOutputProcessing([""], char2idx, pred=True)
    # 학습 과정이 아니므로 디코딩 출력 부분도
    # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)

    targetPredicDec = data.decTargetProcessing([""], char2idx, pred=True)

    for i in range(DEFINES.maxSequenceLength):
        if i > 0:
            outputPredicDec, outputPredicDecLength = data.decOutputProcessing([answer], char2idx, pred=True)
            targetPredicDec = data.decTargetProcessing([answer], char2idx, pred=True)
        # 예측을 하는 부분이다.
        predictions = classifier.predict(
            input_fn=lambda: data.evalInputFn(inputPredicEnc, outputPredicDec, targetPredicDec, 1))

        # 예측한 값을 인지 할 수 있도록
        # 텍스트로 변경하는 부분이다.
        answer = data.pred2string(predictions, idx2char)

        if END in answer:
            break

    print("final answer:", answer)


def main():
    # 데이터를 통한 사전 구성 한다.


    # 테스트용 데이터 만드는 부분이다.
    # 인코딩 부분 만든다.

    # input = ""
    # for i in sys.argv[1:]:
    #     input += i
    #     input += " "

    # print(input)
    # inputPredicEnc, inputPredicEncLength = data.encProcessing([input], char2idx)
    # # 학습 과정이 아니므로 디코딩 입력은
    # # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
    # outputPredicDec, outputPredicDecLength = data.decOutputProcessing([""], char2idx)
    # # 학습 과정이 아니므로 디코딩 출력 부분도
    # # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
    # targetPredicDec = data.decTargetProcessing([""], char2idx)

    # 에스티메이터 구성한다.


    run()


    # 예측을 하는 부분이다.
    # predictions = classifier.predict(
    #     input_fn=lambda:data.evalInputFn(inputPredicEnc, outputPredicDec, targetPredicDec, DEFINES.batchSize))

    # 예측한 값을 인지 할 수 있도록
    # 텍스트로 변경하는 부분이다.
    # data.pred2string(predictions, idx2char)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    # argLength = len(sys.argv)
    #
    # if(argLength < 2):
    #     raise Exception("Don't call us. We'll call you")
    char2idx, idx2char, vocabularyLength = data.loadVocabulary()

    char2idx[PAD] = 0
    char2idx[STD] = 1
    char2idx[END] = 2
    char2idx[UNK] = 3

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
        })

    main()
