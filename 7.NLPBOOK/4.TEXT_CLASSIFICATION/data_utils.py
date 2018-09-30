#Data util for Kaggle dataset

import pandas as pd
from bs4 import BeautifulSoup
import re

bagofwords_path='~/.kaggle/competitions/word2vec-nlp-tutorial/'

#header=0 -> 1번째 줄은 데이터가 아닌 label or label이 없다면 None
#delimiter="\t" -> 데이터들 사이를 구분할 구분자는 탭문자인 "\t"
#quoting=3 -> 따옴표를 문자열을 의미하는 문자가 아닌 일반 문자처럼 사용
#더 잘 쓰기위해서는 pandas 공식 웹페이지에서 각종 옵션들에 대한 설명을 읽는 편이 좋습니다.

# Read data from files 
train = pd.read_csv(bagofwords_path+"labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv(bagofwords_path+"testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv(bagofwords_path+"unlabeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )


def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

    # Download the punkt tokenizer for sentence splitting
import nltk.data
nltk.download()   

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

sentences = []  # Initialize an empty list of sentences

print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)