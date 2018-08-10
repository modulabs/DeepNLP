import unittest
from ngram import nGram
ng = nGram(n=5, corpus_file=None, cache=False)


class TestNgram(unittest.TestCase):
    def test_uni_log(self):
        probability = ng.sentence_probability(sentence='hold your horses', n=1, form='log')
        self.assertAlmostEqual(probability, -24.9337710989)

    def test_uni_antilog(self):
        probability = ng.sentence_probability(sentence='hold your horses', n=1, form='antilog')
        self.assertAlmostEqual(probability, 1.48388689281e-11)

    def test_bi_log(self):
        probability = ng.sentence_probability(sentence='hold your horses', n=2, form='log')
        self.assertAlmostEqual(probability, -18.655540764)

    def test_bi_antilog(self):
        probability = ng.sentence_probability(sentence='hold your horses', n=2, form='antilog')
        self.assertAlmostEqual(probability, 7.90681521418e-09)

    def test_tri_log(self):
        probability = ng.sentence_probability(sentence='hold your horses', n=3, form='log')
        self.assertAlmostEqual(probability, -11.3066636125)

    def test_tri_antilog(self):
        probability = ng.sentence_probability(sentence='hold your horses', n=3, form='antilog')
        self.assertAlmostEqual(probability, 1.22907499816e-05)

    def test_quadri_log(self):
        probability = ng.sentence_probability(sentence='hold your horses', n=4, form='log')
        self.assertAlmostEqual(probability, 0)

    def test_quadri_antilog(self):
        probability = ng.sentence_probability(sentence='hold your horses', n=4, form='antilog')
        self.assertAlmostEqual(probability, 1)

    def test_penti_log(self):
        probability = ng.sentence_probability(sentence='hold your horses', n=5, form='log')
        self.assertAlmostEqual(probability, 0)

    def test_penti_antilog(self):
        probability = ng.sentence_probability(sentence='hold your horses', n=5, form='antilog')
        self.assertAlmostEqual(probability, 1)


if "__name__" == "__main__":
    unittest.main()