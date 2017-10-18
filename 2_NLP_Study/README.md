# Chat_Learning

- NLP_Study Directory 
- 이름 규칙 : (날짜)_(발표주제)_(발표자)
ex) 20170913_Oxford-10강-TTS_강은숙

미시Gan NLP PPT LINK
https://drive.google.com/open?id=0Bx984wTo1QhfVWhIR1R0MWFfZ2M

Watson DeepQA (자연어, 2011년 기준)
왓슨의 핵심인 DeepQA는 ‘11년 퀴즈대회 우승 이후로 현재 의료와 금융에 활용 중

1단계: 질문분석 (소요시간: 0.38sec ) -> 질문을 쪼깨어 단어와 문장의 역할을 나누어 다음과 같은 내용을 분석

1. 어떤 형식의 질문이 들어 온 것인가?
2. 질문의 무엇을 묻는 것인가?
2단계: 가설 생성 (소요시간: 0.52sec) -> 수억개의 문서를 검색하여, 수많은 정답 후보군 (가설)을 생성

3단계: 가설과 증거평가 (소요시간: 0.756sec)

추출한 정답 후보군들 중 명백한 오답을 제거하고, 나머지 후보에서 서로 다른 출처들을 활용하여 해당 구절을 찾아내고, 긍부정 증거를 반영 (Scoring) 평가 알고리즘을 활용하여 관련 정보들의 시간 및 장소 여부로 재 확인함

이제, 필터링을 거친 수백수천개의 정답 후보군에서, slot-grammer parsing, question class detection, parsing term match 등 수백개의 알고리즘을 병행 적용하여 다시 점수를 매김

4단계: 최종 병합 및 점검 (소요시간: 2sec)

활용 알고리즘의 예) Missing-link Detection Answer Lookup Relation Detection Keyword Search Absolute Temporal Match Local-context Passenger Matches Precise Temporal Match Question Class Detection

사람이 경험으로 배우는 것과 유사하게, 왓슨은 자체적으로 수천번의 시뮬레이션으로 지식을 검증하고 결합함 (단, 가장 확률이 높은 답의 정확도가 낫거나, 50%이하이면 대답을 거부함)

https://www.youtube.com/watch?v=HMBPt1g9mak#action=share

왓슨에 대해 공개된 내용 그 이후로 (2011년 제퍼디쇼) 달라진점?

딥러닝 적용

의도분석
분류 (CNN) 등 다양한 분야 적용
