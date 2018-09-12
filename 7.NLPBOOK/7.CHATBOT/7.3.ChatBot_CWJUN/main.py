import tensorflow as tf
import model as ml
import data
import numpy as np
import os

from configs import DEFINES

def main(self):
	inputTrainEncOneHot = []
	outputTrainDecOneHot = []
	# Make Vocabulary
	#######################################################################
	dictionary,  vocabularyLength = data.loadVocabulary()
	# print("#######################################################################")
	# print(dictionary)
	# print(vocabularyLength)
	# print("#######################################################################")
	######################################################################

	# Load Data
	#######################################################################
	xTrain, yTrain, xTest, yTest = data.loadData()
	# print("#######################################################################")
	# print(xTrain)
	# print(yTrain)
	# print("#######################################################################")
	######################################################################

	# START END PADDING  (MaxValue 처리)
	#######################################################################
	inputTrainEnc, inputTrainEncLength = data.encProcessing(xTrain, dictionary)
	outputTrainDec, outputTrainDecLength = data.decOutputProcessing(yTrain, dictionary)
	targetTrainDec = data.decTargetProcessing(yTrain, dictionary)
	
	inputTestEnc, inputTestEncLength = data.encProcessing(xTest,dictionary)
	outputTestDec, outputTestDecLength = data.decOutputProcessing(yTest, dictionary)
	targetTestDec = data.decTargetProcessing(yTest, dictionary)
	# print("#######################################################################")
	# print(type(inputTrainEnc))
	# print(type(outputTrainDec))
	# print(type(targetTrainDec))
	# print(inputTrainEnc.shape)
	# print(outputTrainDec.shape)
	# print(targetTrainDec.shape)
	# print("#######################################################################")
	#######################################################################	
	# classifier 
	#######################################################################
	# Create a directory to save
	checkPointPath = os.path.join(os.getcwd(), DEFINES.checkPointPath)
	os.makedirs(checkPointPath, exist_ok=True)

	classifier = tf.estimator.Estimator(
			model_fn=ml.Model, 
			model_dir=DEFINES.checkPointPath,
			params={
				'hiddenSize': DEFINES.hiddenSize,
				'layerSize': DEFINES.layerSize,
				'learningRate': DEFINES.learningRate,
				'vocabularyLength': vocabularyLength,
				'embeddingSize': DEFINES.embeddingSize,
				'embedding': DEFINES.embedding,
				'embeddingSize': DEFINES.embeddingSize,
				'multilayer': DEFINES.multilayer, 
				'attention': DEFINES.attention,
				'bidirectional': DEFINES.bidirectional,
				'bucket': DEFINES.bucket,
				'beamsearch': DEFINES.beamsearch,
			})
	#print("#######################################################################")
	print(classifier)
	#print("#######################################################################")	
	######################################################################

	#######################################################################
	classifier.train(input_fn=lambda:data.trainInputFn(
		inputTrainEnc, outputTrainDec, targetTrainDec,  DEFINES.batchSize), steps=DEFINES.trainSteps)
	#######################################################################


	#######################################################################
	evalResult = classifier.evaluate(input_fn=lambda:data.evalInputFn(
		inputTestEnc, outputTestDec, targetTestDec,  DEFINES.batchSize))
	print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**evalResult))
	#######################################################################

	# 결과값 바로 나오도록 처리
	# inputPredicEnc, outputPredicDec
	#idx2char, idx2charLength = data.idx2char(dictionary)
	#print(idx2char)
	#print(idx2charLength)
	# enc = 연애를 잘하는사람들 멋져
	inputPredicEnc, inputPredicEncLength = data.encProcessing(["연애를 잘하는사람들 멋져"], dictionary)
	outputPredicDec, outputPredicDecLength = data.decOutputProcessing([""], dictionary)		
	targetPredicDec = data.decTargetProcessing([""], dictionary)		
	#inputPredicEnc, inputPredicEncLength = encProcessing(["연애를 잘하는사람들 멋져"], dictionary)
	print("#######################################################################")	
	print(inputPredicEnc)
	print(outputPredicDec)
	print("#######################################################################")
	predictions = classifier.predict(
		input_fn=lambda:data.evalInputFn(inputPredicEnc, outputPredicDec, targetPredicDec, DEFINES.batchSize))
	print("#######################################################################")	
	print(predictions)
	print("#######################################################################")	
													
if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)
