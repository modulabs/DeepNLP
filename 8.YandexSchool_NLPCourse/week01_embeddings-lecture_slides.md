---
title: "Word Embedding by Yandex"
date: 2017-11-20 15:05:00 +0900
categories: netural language processing
---
아래 내용은 Yandexdataschool / nlp_course / week1_embeddings 내용을 기반으로 정리 하였다.
[참고자료](https://github.com/yandexdataschool/nlp_course/blob/master/resources/slides/lecture1_word_embeddings.pdf)

# Word representation
Word representation 방법은 크게 text, sequence of tokens, vector (word embedding) 과 같이 3가지로 나눌 수 있다.

위 3가지 방법 중 1가지를 선택하여 NLP(Netural Language Processing)의 입력으로 사용한다.

과거의 NLP에서는 주로 2번째인 'sequence of tokens' 방법을 사용했으며 각 토큰에 번호를 부여한 Vocabulary를 구성하고 token 번호를 one-hot vector로 encoding 해서 입력값으로 사용 했다.

><img src="/assets/2018-11-20/yandex_embedding_007.png" />

그런데 위와 같은 방법으로는 hotel과 motel의 유사성을 찾을 수 없다. 가령 사용자가 'Seattle motel'을 찾을 때 'Seattle hotel'의 정보를 제공할 수 없다. 그 이유는 아래와 같다.
1. 모든 one-hot 벡터는 직교한다.
2. one-hot 백터 사이에는 아무런 관계나 유사성이 없다.
3. one-hot 백터는 단어에 관련된 어떠한 정보도 포함하고 있지 않다.

아래와 같은 문맥을 통해서 'bardiwac'란 단어의 뜻을 추론할 수 있다.

><img src="/assets/2018-11-20/yandex_embedding_012.png" />

[Distributional semantics](https://en.wikipedia.org/wiki/Distributional_semantics) 에서는 아래와 같이 표현할 수 있다.

><img src="/assets/2018-11-20/yandex_embedding_013.png" />

위 문장에 다른 단어를 대입할 경우는 아래와 같다. 아래의 내용에서는 'bardiwac'과 'wine'이 가장 유사도가 놉다고 할 수 있다.

><img src="/assets/2018-11-20/yandex_embedding_015.png" />

## 백터의 유사도와 의미의 유사도와 같은가? 이것이 Word embedding의 주된 관심하라 할 수 있다. [John Rupert Firth](https://en.wikipedia.org/wiki/John_Rupert_Firth)

><img src="/assets/2018-11-20/yandex_embedding_018.png" />

# Latent semantic analysis (LSA)
LAS의 기본 아이디어는 특정 단어에 대해 동시에 발생하는 단어의 중복 수를 백터를 표현하는 것이다.

><img src="/assets/2018-11-20/yandex_embedding_022.png" />

아래와 같이 각 단어에 대한 동시에 발생하는 단어의 중복 수 테이블을 특이값 분해를 통해 벡터로 표현할 수 있다. 특이값 분해에 대한 자세한 내용은 '[SVD와 PCA, 그리고 잠재의미분석(LSA)](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/04/06/pcasvdlsa/)'를 참고 하길 바란다.

><img src="/assets/2018-11-20/yandex_embedding_025.png" />

## LSA와 같이 count 기반이 아닌 학습을 통하여 벡터들이 비슷한 의미를 가진 것들은 비슷한 방향을 가르키도록 하는 방법은 없을까?

# [Word2Vec](https://en.wikipedia.org/wiki/Word2vec)

Word2Vec은 아래와 같이 center word와 outside words가 입력(c)에 대해서 출력(o)이 되도록 또는 입력(o)에 대해서 출력(c)가 되도록 학습 한다.

><img src="/assets/2018-11-20/yandex_embedding_029.png" />

window size가 2인 경우는 아래와 같이 center word는 'into' 이며 outside words는 'problems', 'turning', 'banking', 'crises' 4개의 단어가 된다.

><img src="/assets/2018-11-20/yandex_embedding_030.png" />

'into' 다음단어 'banking' 대한 처리도 동일하며 outside words는 'turning', 'banking', 'crises', 'as' 4개의 단어가 된다.

><img src="/assets/2018-11-20/yandex_embedding_031.png" />

목적 함수는 아래와 같이 문장 전체에 대해 center word를 기준으로 outside words가 나올 확률을 높이는 방향으로 학습을 진행 한다.

><img src="/assets/2018-11-20/yandex_embedding_033.png" />

loss 함수는 아래와 같고 loss줄이는 것이 예측 정확도를 높이는 것과 같다.

><img src="/assets/2018-11-20/yandex_embedding_035.png" />

center word에 대해 outer word가 나타날 확률은 다음과 같다.

><img src="/assets/2018-11-20/yandex_embedding_039.png" />

center word 'into'에 대한 outer words 'problems', 'tunning', 'banking', 'crises'의 확률은 아래와 같이 표현할 수 있다.

><img src="/assets/2018-11-20/yandex_embedding_040.png" />

center word와 outer word간의 백터 내적을 크게 하는 것이 활률을 높이는 것이고 이것이 학습 목표이다. 서로의 방향이 유사할 수록 백터의 내적이 커진다.

><img src="/assets/2018-11-20/yandex_embedding_044.png" />

위와 같은 표현은 Deep Learning에서 사용되는 softmax와 유사하다.

><img src="/assets/2018-11-20/yandex_embedding_045.png" />

Θ는 word V에 대한 벡터 이다. 또한 모든 word는 두개의 백터를 갖는다.

><img src="/assets/2018-11-20/yandex_embedding_047.png" />

Skip-Gram은 주어진 center word에 대해서 outside words를 예측하는 방법이다.

><img src="/assets/2018-11-20/yandex_embedding_048.png" />

CBOW (Continuous Bag of Wrods)는 outsied words로부터 center word를 예측하는 방법이다.

><img src="/assets/2018-11-20/yandex_embedding_049.png" />

학습해야할 데이터가 많을경우 연산시간은 V 에 비례한다. 이를 줄이기위한 방법으로 '[Hierarchical softmax](http://dalpo0814.tistory.com/7)', '[Negative sampling](http://dalpo0814.tistory.com/6)' 방법이 있다.

><img src="/assets/2018-11-20/yandex_embedding_052.png" />

Word2Vec는 [Matrix factorization](http://sanghyukchun.github.io/73/)과 동등하다.

><img src="/assets/2018-11-20/yandex_embedding_054.png" />

# [GloVe](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/04/09/glove/): combile count-based and direct prediction methods

GloVe의 수식은 아래와 같다.

><img src="/assets/2018-11-20/yandex_embedding_056.png" />

실제 값과 예측값 사이에 오차 제곱을 하는 것은 LSA와 비슷하다.

><img src="/assets/2018-11-20/yandex_embedding_057.png" />

그리고 noise를 제거하기 위한 함수를 추가 했다. 특정 단어가 너무 많이 발생할 경우 이를 최대값 까지만 사용하겠다는 의미 이다.

><img src="/assets/2018-11-20/yandex_embedding_058.png" />

# Relations between vectors

Word2Vec에서 백터(king)에서 백터(man)을 빼고 백터(woman)을 더하면 백터(queen)과 비슷한 값이 나온다.

><img src="/assets/2018-11-20/yandex_embedding_059.png" />

나라백터 수도백터 관계는 아래와 같다.

><img src="/assets/2018-11-20/yandex_embedding_060.png" />

형용사에 대한 비교급/최상위급 관계는 아라와 같다.

><img src="/assets/2018-11-20/yandex_embedding_061.png" />

개구리(flog)의 인접 단어들은 아래와 같다.

><img src="/assets/2018-11-20/yandex_embedding_062.png" />

Symantic space에서 보면 아래와 같다.

><img src="/assets/2018-11-20/yandex_embedding_066.png" />

><img src="/assets/2018-11-20/yandex_embedding_067.png" />

><img src="/assets/2018-11-20/yandex_embedding_066.png" />

><img src="/assets/2018-11-20/yandex_embedding_069.png" />

><img src="/assets/2018-11-20/yandex_embedding_070.png" />

><img src="/assets/2018-11-20/yandex_embedding_071.png" />

><img src="/assets/2018-11-20/yandex_embedding_072.png" />

><img src="/assets/2018-11-20/yandex_embedding_073.png" />

><img src="/assets/2018-11-20/yandex_embedding_074.png" />

><img src="/assets/2018-11-20/yandex_embedding_075.png" />

><img src="/assets/2018-11-20/yandex_embedding_076.png" />

Embedding을 평가하는 방법은 아래와 같다. 여러 embedding을 동일한 모델에 적용해 보고 결과가 더 좋을 embedding을 더 좋다고 평가한다.

><img src="/assets/2018-11-20/yandex_embedding_077.png" />

# FastText - Adding subwrod information
* Skip gram / negative sampling을 통해 모델을 구성한다.
* n-gram을 통해 단어를 나눈 후 모든 n-gram을 합한것을 단어의 백터로 사용한다.›

><img src="/assets/2018-11-20/yandex_embedding_079.png" />

기존 embedding을 이용해 새로운 embedding을 만드는 방법은 아래와 같다.

><img src="/assets/2018-11-20/yandex_embedding_080.png" />

><img src="/assets/2018-11-20/yandex_embedding_081.png" />

><img src="/assets/2018-11-20/yandex_embedding_082.png" />

# Skip-gram model to the sentence level

단어가 아닌 문장을 encoding 하는 방법을 사용한다.

><img src="/assets/2018-11-20/yandex_embedding_084.png" />

><img src="/assets/2018-11-20/yandex_embedding_085.png" />

Machine Translation에서는 아래와 같다.

><img src="/assets/2018-11-20/yandex_embedding_087.png" />

Machine Translation에서 번역할 언어의 정보가 없다면 어떻게 할 것인가?

><img src="/assets/2018-11-20/yandex_embedding_089.png" />

><img src="/assets/2018-11-20/yandex_embedding_090.png" />

결론은 다음곽 같다.

><img src="/assets/2018-11-20/yandex_embedding_092.png" />

><img src="/assets/2018-11-20/yandex_embedding_093.png" />

><img src="/assets/2018-11-20/yandex_embedding_094.png" />

