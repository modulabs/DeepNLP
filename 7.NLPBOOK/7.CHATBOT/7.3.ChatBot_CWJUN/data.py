import pandas as pd
import tensorflow as tf
import enum
import os
import re
from sklearn.model_selection import train_test_split
import numpy as np
from configs import DEFINES

FILTERS = "([~.,!?\"':;)(])"
PAD = "<PADDING>"
STD = "<START>"
END = "<END>"
UNK = "<UNKWON>"

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
changeFilter = re.compile(FILTERS)

def loadData():
	dataDF = pd.read_csv(DEFINES.dataPath, header=0)
	question, answer = list(dataDF['Q']), list(dataDF['A'])
	xTrain, xTest, yTrain, yTest = train_test_split(question, answer, test_size=0.33, random_state=42)
	return xTrain, yTrain, xTest, yTest

def encProcessing(xValue, dictionary):
	sequencesInputIndex  = []
	sequencesLength = []
	for sequence in xValue:
		sequence = re.sub(changeFilter, "", sequence)
		sequenceIndex = []
		for word in sequence.split():
			if dictionary.get(word) is not None:
				sequenceIndex.extend([dictionary.get(word)])
			else:
				sequenceIndex.extend([dictionary.get('<UNKWON>')])
		sequencesLength.append(len(sequenceIndex))
		sequenceIndex += (DEFINES.maxSequenceLength - len(sequenceIndex)) * [dictionary.get('<PADDING>')]
		sequencesInputIndex.append(sequenceIndex)

	return np.asarray(sequencesInputIndex), sequencesLength

def decOutputProcessing(yValue, dictionary):
	sequencesOutputIndex = []
	sequencesLength = []

	for i, sequence in enumerate(yValue):
		sequence = re.sub(changeFilter, "", sequence)
		sequenceIndex = []
		sequenceIndex = [dictionary.get('<START>')] + [dictionary.get(word) for word in sequence.split()]
		sequencesLength.append(len(sequenceIndex))
		sequenceIndex += (DEFINES.maxSequenceLength - len(sequenceIndex)) * [dictionary.get('<PADDING>')]
		# if len(sequenceIndex) != 15:
		# 	print(i, ':', sequenceIndex, '::', sequence)
		sequencesOutputIndex.append(sequenceIndex)

	return np.asarray(sequencesOutputIndex), sequencesLength

def decTargetProcessing(yValue, dictionary):
	sequencesTargetIndex = []

	for sequence in yValue:
		sequence = re.sub(changeFilter, "", sequence)
		sequenceIndex = [dictionary.get(word) for word in sequence.split()] + [dictionary.get('<END>')]
		sequenceIndex += (DEFINES.maxSequenceLength - len(sequenceIndex)) * [dictionary.get('<PADDING>')]
		sequencesTargetIndex.append(sequenceIndex)

	return np.asarray(sequencesTargetIndex)

def rearrange(input, output, target):
	features = {"input": input, "output": output}
	return features, target 

#encoder decoder
def trainInputFn(inputTrainEnc, outputTrainDec, targetTrainDec, batchSize):
	dataset = tf.data.Dataset.from_tensor_slices((inputTrainEnc, outputTrainDec, targetTrainDec))
	dataset = dataset.shuffle(buffer_size=len(inputTrainEnc))
	assert batchSize is not None, "train batchSize must not be None"
	dataset = dataset.batch(batchSize)
	dataset = dataset.map(rearrange)
	dataset = dataset.repeat()
	iterator = dataset.make_one_shot_iterator()
	return iterator.get_next()

def evalInputFn(inputTrainEnc, outputTrainDec, targetTrainDec, batchSize):
	dataset = tf.data.Dataset.from_tensor_slices((inputTrainEnc, outputTrainDec, targetTrainDec))
	dataset = dataset.shuffle(buffer_size=len(inputTrainEnc))
	assert batchSize is not None, "eval batchSize must not be None"
	dataset = dataset.batch(batchSize)
	dataset = dataset.map(rearrange)
	dataset = dataset.repeat(1)
	iterator = dataset.make_one_shot_iterator()
	return iterator.get_next()

def dataTokenizer(data):
	words = []
	for sentence in data:
		sentence =re.sub(changeFilter, "", sentence)
		for word in sentence.split():
			words.append(word) 

	return [word for word in words if word]

def loadVocabulary():
	vocabularyList = []
	if(not(os.path.exists(DEFINES.vocabularyPath))):
		dataDF = pd.read_csv(DEFINES.dataPath, encoding='utf-8')
		question, answer = list(dataDF['Q']), list(dataDF['A'])
		if(os.path.exists(DEFINES.dataPath)):
			data = []
			data.extend(question)
			data.extend(answer)
			words = dataTokenizer(data)
			words = list(set(words))
			# Add PreDefine 
			[words.insert(0, word) for word in reversed(MARKER)]
			# print(words)	
		with open(DEFINES.vocabularyPath, 'w') as vocabularyFile:
			for word in words:
				vocabularyFile.write(word + '\n')

	with open(DEFINES.vocabularyPath, 'r', encoding='utf-8') as vocabularyFile:
		for line in vocabularyFile:
			vocabularyList.append(line.strip())

	index = char2idx(vocabularyList)
	return index, len(index) 

def char2idx(vocabularyList):
	char2idx = {char: idx for idx, char in enumerate(vocabularyList)}
	#print(char2idx)
	#print(len(char2idx))
	return char2idx

def idx2char(dictionary):
	idx2char = {idx: char for idx, char in enumerate(dictionary)}
	#print(idx2char)
	#print(len(idx2char))
	return char, len(char) 

def main(self):
	dictionary, vocabularyLength = loadVocabulary()	
	#print(dictionary)
	#print(vocabularyLength)
	#xTrain, yTrain, xTest, yTest = loadData()
	#print(xTrain)
	#print(dictionary)
	#inputTrainEnc, inputTrainEncLength = encProcessing(xTrain, dictionary)
	#inputTestEnc, inputTestEncLength = encProcessing(xTest, dictionary)
	#outputTrainDec, outputTrainDecLength = decOutputProcessing(yTrain, dictionary)
	#outputTestDec, outputTestDecLength = decOutputProcessing(yTest, dictionary)
	#targetTrainDec = decTargetProcessing(yTrain, dictionary)
	#targetTestDec = decTargetProcessing(yTest, dictionary)	

	inputPredicEnc, inputPredicEncLength = encProcessing(["연애를 잘하는사람들 멋져"], dictionary)
	print(inputPredicEnc)
	print(inputPredicEncLength)
	outputPredicDec, outputPredicDecLength = decOutputProcessing([""], dictionary)				
	print(outputPredicDec)
	print(outputPredicDecLength)

if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)
