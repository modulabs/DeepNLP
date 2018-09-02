import pandas as pd
import tensorflow as tf
import enum
import os
import re
from sklearn.model_selection import train_test_split
import numpy as np
from configs import DEFINES

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing import sequence

PAD = "<PADDING>"
STD = "<START>"
END = "<END>"
UNK = "<UNKWON>"

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]

mVocabularyList = []
mVocabularyDictionary = []
mVocabularyLength = 0

tokenizer = Tokenizer()

def loadData():
	data = pd.read_csv(DEFINES.dataPath, header=0)
	changeMark = re.compile("([~.,!?\"':;)(012])")
	mX = []
	mY = []
	X = data.pop('Q');
	Y = data.pop('A');
	for x in X:
		mX.append(re.sub(changeMark, "", x))
	for y in Y:
		mY.append(re.sub(changeMark, "", y))

	xTrain, xTest, yTrain, yTest = train_test_split(mX, mY, test_size=0.33, random_state=42)
	return xTrain, yTrain, xTest, yTest

def preProcessing(xTrain, yTrain, xTest, yTest):
	xTrain = tokenizer.texts_to_sequences(xTrain)
	yTrain = tokenizer.texts_to_sequences(yTrain)
	xTest = tokenizer.texts_to_sequences(xTest)
	yTest = tokenizer.texts_to_sequences(yTest)
	xTrain = pad_sequences(xTrain, maxlen=DEFINES.maxSequenceLength, value=PAD_INDEX)	
	yTrain = pad_sequences(yTrain, maxlen=DEFINES.maxSequenceLength, value=PAD_INDEX)	
	xTest = pad_sequences(xTest, maxlen=DEFINES.maxSequenceLength, value=PAD_INDEX)	
	yTest = pad_sequences(yTest, maxlen=DEFINES.maxSequenceLength, value=PAD_INDEX)	
	return xTrain, yTrain, xTest, yTest


def trainInputFn(features, labels, batchSize):
	dataset = tf.data.Dataset.from_tensor_slices((features, labels))
	dataset = dataset.shuffle(DEFINES.shuffleSeek).repeat().batch(batchSize)
	return dataset

def evalInputFn(features, labels, batchSize):
	if labels is None:
		inputs = features
	else:
		inputs = (features, labels)

	dataset = tf.data.Dataset.from_tensor_flices(inputs)
	assert batchSize is not None, "batchSize must not be None"
	dataset = dataset.batch(batchSize)
	return dataset

def dataTokenizer(sentence):
	words = []
	changeMark = re.compile("([.,!?\"':;)(012])")
	sentence = re.sub(changeMark, "", sentence)

	for word in sentence.strip().split():
		#print(word)
		words.append(word)

	return [word for word in words if word]

def loadVocabulary():
	vocabularyList = MARKER + []

	if(not(os.path.exists(DEFINES.vocabularyPath))):
		with open(DEFINES.dataPath, 'r', encoding='utf-8') as dataFile:
			data = dataFile.read()
			words = dataTokenizer(data)
			words = list(set(words))
		with open(DEFINES.vocabularyPath, 'w') as vocabularyFile:
			for word in words:
				vocabularyFile.write(word + '\n')

	with open(DEFINES.vocabularyPath, 'r', encoding='utf-8') as vocabularyFile:
		for line in vocabularyFile:
			vocabularyList.append(line.strip())
	
	vocabularyDictionary = tokenizer.fit_on_texts(vocabularyList)
	wordVocab = tokenizer.word_index
	#print(wordVocab)
	print("================================")
	return wordVocab, len(wordVocab) 

# def indexsToWords(indexs):
# 	words = []
# 	for index in indexs:
# 		words.append(mVocabularyDictionary[index])
# 	return words

# def wordsToIndexs(words):
# 	indexs = []
# 	for word in words:
# 		if word in mVocabularyDictionary:
# 			indexs.append(mVocabularyDictionary[word])
# 		else:
# 			indexs.append(UNK_INDEX)
# 	return indexs



def main(self):
	vocabularyDictionary, vocabularyLength = loadVocabulary()	
	xTrain, yTrain, xTest, yTest = loadData()
	#print(vocabularyDictionary)
	print("================================")
	#print(vocabularyLength)
	print("================================")
	print(xTrain)
	print("================================")
	train_text_sequences = tokenizer.texts_to_sequences(xTrain)
	#indexs = wordsToIndexs(xTrain)
	#print(train_text_sequences)
	print("================================")

	train_text_sequences = pad_sequences(train_text_sequences, maxlen=DEFINES.maxSequenceLength, value=PAD_INDEX)
	print(train_text_sequences)

if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)
