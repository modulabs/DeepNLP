import  string
from    nltk.tokenize import word_tokenize
from    nltk.corpus import stopwords
import  nltk
import  os
import  re

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

# 2. pos tagging 후 필요 없는 부분 삭제 (nltk, 형태소 분석
#     (1) 추후에 필요시 적용

# paper, label index 로 변경

# word2vec, glove


paperdirectory = "../data_title/abstract"
labeldirectory = "../data_title/title"
vocab_path = "../data_title/vocab"


def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words

def load_vocab(vocab_path):
    # print("load vocab ...")
    with open(vocab_path, 'rb') as f:
        words = f.read().decode('utf-8').splitlines()
    return {words[i]: i for i in range(len(words))}


def sentence2id(_vocab, line):
    return [_vocab.get(token, _vocab['<unk>']) for token in basic_tokenizer(line)]


def deleteStopword(text):
    return basic_tokenizer(text)

def deleteUnnecessaryPOS(words):
    # pos tagging
    # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

    tags = nltk.pos_tag(words)
    print(tags)

    # 필요없는 부분 삭제 후 words 로 리턴
    return words




def ShortenedRecovery(text):
    return text


def Processing(filename):
    file = open(filename, 'rt')
    text = file.read()
    file.close()

    text = ShortenedRecovery(text)
    words = deleteStopword(text)
    # words = deleteFrequency(words)
    # words = deleteUnnecessaryPOS(words)

    return words


def saveVocab(words):
    word_threshold = 2

    sorted_vocab = sorted(words, key=vocab.get, reverse=True)

    with open(vocab_path, 'wb') as f:
        f.write(('<pad>' + '\n').encode('utf-8'))
        f.write(('<unk>' + '\n').encode('utf-8'))
        f.write(('<s>' + '\n').encode('utf-8'))
        f.write(('<\s>' + '\n').encode('utf-8'))
        index = 4
        for word in sorted_vocab:
            if vocab[word] < word_threshold:
                pass

            if len(word) <= 1:
                continue

            f.write((word + '\n').encode('utf-8'))
            index += 1


def makeDic(filename, vocab):
    words = Processing(filename)

    for token in words:
        if not token in vocab:
            vocab[token] = 0
        vocab[token] += 1

    return


vocab = {}

for root, dirs, files in os.walk(paperdirectory):
    for filename in files:
        # print(paperdirectory + "/" + filename)
        makeDic(paperdirectory + "/" + filename, vocab)


for root, dirs, files in os.walk(labeldirectory):
    for filename in files:
        # print(labeldirectory + "/" + filename)
        makeDic(labeldirectory + "/" + filename, vocab)

print("======== vocab ============")
print(len(vocab))  # 23,742

saveVocab(vocab)
# paper, label index 로 변경





def token2id(  data, mode):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """

    in_path = data
    out_path = data + '_ids.' + mode

    in_file = open(in_path, 'rb')
    out_file = open(out_path, 'wb')
    _vocab = load_vocab(vocab_path)
    print(_vocab)

    lines = in_file.read().decode('utf-8').splitlines()
    for line in lines:
        if mode == 'dec':  # we only care about '<s>' and </s> in decoder
            ids = [_vocab['<s>']]
        else:
            ids = []

        sentence_ids = sentence2id(_vocab, line)
        ids.extend(sentence_ids)
        if mode == 'dec':
            ids.append(_vocab['<\s>'])

        out_file.write(b' '.join(str(id_).encode('cp1252') for id_ in ids) + b'\n')



for root, dirs, files in os.walk(paperdirectory):
    for filename in files:
        if not '_ids' in filename:
            token2id( paperdirectory + "/" + filename, 'enc')


for root, dirs, files in os.walk(labeldirectory):
    for filename in files:
        if not '_ids' in filename:
            token2id( labeldirectory + "/" + filename, 'dec')