import  string
from    nltk.tokenize import word_tokenize
from    nltk.corpus import stopwords
import  nltk
import  os

# 사전구축
# 0. 축약어복원
#     (1) 주어와 be동사를 i'm -> i am, you're -> You are, he's -> he is, that's -> that is
#     (2) 주어와 have, has 동사를  I've -> i have, he's -> he has
#     (3) 주어와 조동사를  you'll -> you will, she'll -> she will, it'll -> it will,
#         she'd -> she would, we'd -> we had, they'd -> they should
#     (4) 의문사와 be동사나 조동사
#         how's -> how is, what's -> what is, who'll -> who will
#     (5) be동사와 not
#         isn't -> is not, aren't -> are not, wasn't -> was not, weren't -> were not
#     (6) do와 not
#         don't -> do not, doesn't -> does not, didn't -> did not
#     (7) have 와 not
#         have't -> have not, hasn't -> has not
#     (8) 조동사와 not
#         won't -> will not, couldn't -> could not, shouldn't -> should not
#         mightn't -> might not, mustn't -> must not
#     (9) 조동사와 have 동사 축약
#         would've -> would have


# 1. stop word 제거
#     (1).   remove punctuation from each word =>  !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
#     (2).   remove remaining tokens that are not alphabetic
#     (3).   filter out stop words ('english')
#     (4).   word len 도 제한 필요  (너무 짧거나, 길면 제외 : 기준필요)
#     (5).   빈도수가 적은 word는 삭제

# 2. pos tagging 후 필요 없는 부분 삭제
#     (1) 추후에 필요시 적용

# paper, label index 로 변경

# word2vec, glove


paperdirectory = "../data/paper"
labeldirectory = "../data/label"


def deleteStopword(filename):
    file = open(filename, 'rt')
    text = file.read()
    file.close()

    # split into words
    tokens = word_tokenize(text)

    # convert to lower case
    tokens = [w.lower() for w in tokens]

    # remove punctuation from each word =>  !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]

    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]

    print(words)
    return words


def deleteUnnecessaryPOS(words):
    # pos tagging
    # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

    tags = nltk.pos_tag(words)
    print(tags)

    return tags


def deleteFrequency(words):
    return


def makeDic(filename):
    words = deleteStopword(filename)
    # words = deleteUnnecessaryPOS(words)
    words = deleteFrequency(words)
    # saveDic(words)


for root, dirs, files in os.walk(paperdirectory):
    for filename in files:
        print(paperdirectory + "/" + filename)
        makeDic(paperdirectory + "/" + filename)


for root, dirs, files in os.walk(labeldirectory):
    for filename in files:
        print(labeldirectory + "/" + filename)
        makeDic(labeldirectory + "/" + filename)





