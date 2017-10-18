import math
import nltk
import time
import collections

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_c = collections.defaultdict(int)
    bigram_c = collections.defaultdict(int)
    trigram_c = collections.defaultdict(int)

    for sentence in training_corpus:
        tokens0 = sentence.strip().split()
        tokens1 = tokens0 + [STOP_SYMBOL]
        tokens2 = [START_SYMBOL] + tokens0 + [STOP_SYMBOL]
        tokens3 = [START_SYMBOL] + [START_SYMBOL] + tokens0 + [STOP_SYMBOL]
        # unigrams
        for unigram in tokens1:
            unigram_c[unigram] += 1

        # bigrams
        for bigram in nltk.bigrams(tokens2):
            bigram_c[bigram] += 1

        # trigrams
        for trigram in nltk.trigrams(tokens3):
            trigram_c[trigram] += 1

    unigrams_len = sum(unigram_c.itervalues())
    unigram_p = {k: math.log(float(v) / unigrams_len, 2) for k, v in unigram_c.iteritems()}

    # calc P(W2|W1) = P(W2,W1) / P(W1) = C(W2,W1) / C(W1)
    unigram_c[START_SYMBOL] = len(training_corpus)
    bigram_p = {k: math.log(float(v) / unigram_c[k[0]], 2) for k, v in bigram_c.iteritems()}

    bigram_c[(START_SYMBOL, START_SYMBOL)] = len(training_corpus)
    trigram_p = {k: math.log(float(v) / bigram_c[k[:2]], 2) for k, v in trigram_c.iteritems()}


# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc.
def score(ngram_p, n, corpus):
    scores = []
    for sentence in corpus:
        sentence_score = 0
        tokens0 = sentence.strip().split()
        if n == 1:
            tokens = tokens0 + [STOP_SYMBOL]
        elif n == 2:
            tokens = nltk.bigrams([START_SYMBOL] + tokens0 + [STOP_SYMBOL])
        elif n == 3:
            tokens = nltk.trigrams([START_SYMBOL] + [START_SYMBOL] + tokens0 + [STOP_SYMBOL])
        else:
            raise ValueError('Parameter "n" has an invalid value %s' % n)
        for token in tokens:
            try:
                p = ngram_p[token]
            except KeyError:
                p = MINUS_INFINITY_SENTENCE_LOG_PROB
            sentence_score += p
        scores.append(sentence_score)
    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    return scores

def linearscore(unigrams, bigrams, trigrams, corpus):
    """Linear interpolate the probabilities.

    See http://web.stanford.edu/~jurafsky/slp3/4.pdf paragraph 4.4.3
    """

    scores = []
    # Set lambda equal to all the n-grams so that it sums up to 1.
    lambda_ = 1.0 / 3
    for sentence in corpus:
        interpolated_score = 0
        tokens0 = sentence.strip().split()
        for trigram in nltk.trigrams([START_SYMBOL] + [START_SYMBOL] + tokens0 + [STOP_SYMBOL]):
            try:
                p3 = trigrams[trigram]
            except KeyError:
                p3 = MINUS_INFINITY_SENTENCE_LOG_PROB
            try:
                p2 = bigrams[trigram[1:3]]
            except KeyError:
                p2 = MINUS_INFINITY_SENTENCE_LOG_PROB
            try:
                p1 = unigrams[trigram[2]]
            except KeyError:
                p1 = MINUS_INFINITY_SENTENCE_LOG_PROB
            interpolated_score += math.log(lambda_ * (2 ** p3) + lambda_ * (2 ** p2) + lambda_ * (2 ** p1), 2)
        scores.append(interpolated_score)
    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close()

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
