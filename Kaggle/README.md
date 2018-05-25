<p>
NLP 관련 Kaggle 후보군 정리

<p>
Title: Sentiment Analysis on Movie Reviews
Kaggle : https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews
Data Set : Rotten Tomatoes dataset
주제 : 감정 분석 5가지 감정 분석 
0- negative
1 - somewhat negative
2 - neutral
3 - somewhat positive
4 - positive

<p>
Title : Toxic Comment Classification Challenge
Kaggle : https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
주제 : 악플 탐지

<p>
Title : Spooky Author Identification
Kaggle : https://www.kaggle.com/c/spooky-author-identification
주제 : 작가의 스타일 찾기

<p>
Title: Bag of Words Meets Bags of Popcorn
Kaggle : https://www.kaggle.com/c/word2vec-nlp-tutorial
Data Set : IMDB dataset
주제 : 영화 리뷰 보고 긍정 부정 분석

<p>
Title: What's Cooking?
Kaggle : https://www.kaggle.com/c/whats-cooking
주제 : 감정 분석





- 카글 튜토리얼 : DeepLearningMovies https://github.com/wendykan/DeepLearningMovies
>  25,000개의 영화 리뷰 댓글 내용과 그에 딸린 추천/비추천이 일치하는지, 그 정확도를 두고 호불호를 예측하는 모형
- 
>  워드백 모형이란 ? 
- 
>  Word2Vec의 장점 
* [ ]  많은 양의 데이터를 End2End로 학습시키기 때문에 라벨데이터가 필요 없음
* [ ]  기존 RNN모델 대비 학습시간을 크게 단축시킴
* [ ]  비슷한 의미를 가진 단어를 군집하며, 거리를 기준으로 유사도를 숫자로 비교할 수 있음

>  구조
* [ ]  BagOfWords.py - 머신러닝 예측모델 생성(랜덤포레스트분류기/앙상블모델)
* [ ]  KaggleWord2VecUtility.py - 댓글 수집 및 전처리
* [ ]  Word2Vec_AverageVectors.py - 벡터 길이를 맞춰주기 위해 300차원 단어 벡터에 대한 평균화 진행, 하지만 Word Bag보다 성능이 떨어짐. 
* [ ]  Word2Vec_BagOfCentroids.py - 이를 개선하기 위해 scikit-learn에서 TfidfVectorizer를 통해 tf-idf 가중치를 사용해 성능향상을 시도
>  사전 설치 라이브러리
* [ ]  numpy==1.9.2
* [ ]  scipy
* [ ]  scikit-learn
* [ ]  nltk
* [ ]  pandas==0.16.0
* [ ]  beautifulsoup4

>  발표영상
- 4월 25일

  * 전창욱님: https://youtu.be/cICsqOzpVLk
  
  * 최태균님: https://youtu.be/raE_rp_YuY0

- 5월 2일

  * 최태균님: https://youtu.be/1CtCJwGZICE
  
- 5월 16일

  * 최태균님: https://youtu.be/ZhTr5M7AmEY
  
- 5월 23일
   
  * 전창욱님: https://youtu.be/ztuu5xc7_M0
  
