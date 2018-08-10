from __future__ import division
from collections import Counter
import math as calc


class nGram():
    """A program which creates n-Gram (1-5) Maximum Likelihood Probabilistic Language Model with Laplace Add-1 smoothing
    and stores it in hash-able dictionary form.
    n: number of bigrams (supports up to 5)
    corpus_file: relative path to the corpus file.
    cache: saves computed values if True


Usage:
>>> ng = nGram(n=5, corpus_file=None, cache=False)
>>> print(ng.sentence_probability(sentence='hold your horses', n=2, form='log'))
>>> -18.655540764
"""
    def __init__(self, n=1, corpus_file=None, cache=False):
        """Constructor method which loads the corpus from file and creates ngrams based on imput parameters."""
        self.words = []
        self.load_corpus(corpus_file)
        self.unigram = self.bigram = self.trigram = self.quadrigram = self.pentigram = None
        self.create_unigram(cache)
        if n >= 2:
            self.create_bigram(cache)
        if n >= 3:
            self.create_trigram(cache)
        if n >= 4:
            self.create_quadrigram(cache)
        if n >= 5:
            self.create_pentigram(cache)
        return

    def load_corpus(self, file_name):
        """Method to load external file which contains raw corpus."""
        print("Loading Corpus from data file")
        if file_name is None:
            file_name = "corpus.data"
        corpus_file = open(file_name, 'r')
        corpus = corpus_file.read()
        corpus_file.close()
        print("Processing Corpus")
        self.words = corpus.split(' ')
    
    def create_unigram(self, cache):
        """Method to create Unigram Model for words loaded from corpus."""
        print("Creating Unigram Model")
        unigram_file = None
        if cache:
            unigram_file = open('unigram.data', 'w')
        print("Calculating Count for Unigram Model")
        unigram = Counter(self.words)
        if cache:
            unigram_file.write(str(unigram))
            unigram_file.close()
        self.unigram = unigram

    def create_bigram(self, cache):
        """Method to create Bigram Model for words loaded from corpus."""
        print("Creating Bigram Model")
        words = self.words
        biwords = []
        for index, item in enumerate(words):
            if index == len(words)-1:
                break
            biwords.append(item+' '+words[index+1])
        print("Calculating Count for Bigram Model")
        bigram_file = None
        if cache:
            bigram_file = open('bigram.data', 'w')
        bigram = Counter(biwords)
        if cache:
            bigram_file.write(str(bigram))
            bigram_file.close()
        self.bigram = bigram

    def create_trigram(self, cache):
        """Method to create Trigram Model for words loaded from corpus."""
        print("Creating Trigram Model")
        words = self.words
        triwords = []
        for index, item in enumerate(words):
            if index == len(words)-2:
                break
            triwords.append(item+' '+words[index+1]+' '+words[index+2])
        print("Calculating Count for Trigram Model")
        if cache:
            trigram_file = open('trigram.data', 'w')
        trigram = Counter(triwords)
        if cache:
            trigram_file.write(str(trigram))
            trigram_file.close()
        self.trigram = trigram

    def create_quadrigram(self, cache):
        """Method to create Quadrigram Model for words loaded from corpus."""
        print("Creating Quadrigram Model")
        words = self.words
        quadriwords = []
        for index, item in enumerate(words):
            if index == len(words)-3:
                break
            quadriwords.append(item+' '+words[index+1]+' '+words[index+2]+' '+words[index+3])
        print("Calculating Count for Quadrigram Model")
        if cache:
            quadrigram_file = open('fourgram.data', 'w')
        quadrigram = Counter(quadriwords)
        if cache:
            quadrigram_file.write(str(quadrigram))
            quadrigram_file.close()
        self.quadrigram = quadrigram

    def create_pentigram(self, cache):
        """Method to create Pentigram Model for words loaded from corpus."""
        print("Creating pentigram Model")
        words = self.words
        pentiwords = []
        for index, item in enumerate(words):
            if index == len(words)-4:
                break
            pentiwords.append(item+' '+words[index+1]+' '+words[index+2]+' '+words[index+3]+' '+words[index+4])
        print("Calculating Count for pentigram Model")
        if cache:
            pentigram_file = open('pentagram.data', 'w')
        pentigram = Counter(pentiwords)
        if cache:
            pentigram_file.write(str(pentigram))
            pentigram_file.close()
        self.pentigram = pentigram

    def probability(self, word, words="", n=1):
        """Method to calculate the Maximum Likelihood Probability of n-Grams on the basis of various parameters."""
        if n == 1:
            return calc.log((self.unigram[word]+1)/(len(self.words)+len(self.unigram)))
        elif n == 2:
            return calc.log((self.bigram[words]+1)/(self.unigram[word]+len(self.unigram)))
        elif n == 3:
            return calc.log((self.trigram[words]+1)/(self.bigram[word]+len(self.unigram)))
        elif n == 4:
            return calc.log((self.quadrigram[words]+1)/(self.trigram[word]+len(self.unigram)))
        elif n == 5:
            return calc.log((self.pentigram[words]+1)/(self.quadrigram[word]+len(self.unigram)))

    def sentence_probability(self, sentence, n=1, form='antilog'):
        """Method to calculate cumulative n-gram Maximum Likelihood Probability of a phrase or sentence."""
        words = sentence.lower().split()
        P = 0
        if n == 1:
            for index, item in enumerate(words):
                P += self.probability(item)
        if n == 2:
            for index, item in enumerate(words):
                if index >= len(words) - 1:
                    break
                P += self.probability(item, item+' '+words[index+1], 2)
        if n == 3:
            for index, item in enumerate(words):
                if index >= len(words) - 2:
                    break
                P += self.probability(item+' '+words[index+1], item+' '+words[index+1]+' '+words[index+2], 3)
        if n == 4:
            for index, item in enumerate(words):
                if index >= len(words) - 3:
                    break
                P += self.probability(item+' '+words[index+1]+' '+words[index+2], item+' '+words[index+1]+' ' +
                                      words[index+2]+' '+words[index+3], 4)
        if n == 5:
            for index, item in enumerate(words):
                if index >= len(words) - 4:
                    break
                P += self.probability(item+' '+words[index+1]+' '+words[index+2]+' '+words[index+3], item+' ' +
                                      words[index+1]+' '+words[index+2]+' '+words[index+3]+' '+words[index+4], 5)
        if form == 'log':
            return P
        elif form == 'antilog':
            return calc.pow(calc.e, P)

help(nGram)
