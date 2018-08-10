[![Build Status](https://travis-ci.org/jbhoosreddy/ngram.svg?branch=master)](https://travis-ci.org/jbhoosreddy/ngram)

Ngram
=====

A software which creates n-Gram (1-5) Maximum Likelihood Probabilistic Language Model with Laplace Add-1 smoothing and stores it in hash-able dictionary form.

    class nGram
        |  A program which creates n-Gram (1-5) Maximum Likelihood Probabilistic Language Model with Laplace Add-1 smoothing
        |      and stores it in hash-able dictionary form.
        |      n: number of bigrams (supports up to 5)
        |      corpus_file: relative path to the corpus file.
        |      cache: saves computed values if True
        |
        |
        |  Usage:
        |  >>> ng = nGram(n=5, corpus_file=None, cache=False)
        |  >>> print(ng.sentence_probability(sentence='hold your horses', n=2, form='log'))
        |  >>> -18.655540764
        |
        |  Methods defined here:
        |
        |  __init__(self, n=1, corpus_file=None, cache=False)
        |      Constructor method which loads the corpus from file and creates ngrams based on imput parameters.
        |
        |  create_bigram(self, cache)
        |      Method to create Bigram Model for words loaded from corpus.
        |
        |  create_pentigram(self, cache)
        |      Method to create Pentigram Model for words loaded from corpus.
        |
        |  create_quadrigram(self, cache)
        |      Method to create Quadrigram Model for words loaded from corpus.
        |
        |  create_trigram(self, cache)
        |      Method to create Trigram Model for words loaded from corpus.
        |
        |  create_unigram(self, cache)
        |      Method to create Unigram Model for words loaded from corpus.
        |
        |  load_corpus(self, file_name)
        |      Method to load external file which contains raw corpus.
        |
        |  probability(self, word, words='', n=1)
        |      Method to calculate the Maximum Likelihood Probability of n-Grams on the basis of various parameters.
        |
        |  sentence_probability(self, sentence, n=1, form='antilog')
        |      Method to calculate cumulative n-gram Maximum Likelihood Probability of a phrase or sentence.
