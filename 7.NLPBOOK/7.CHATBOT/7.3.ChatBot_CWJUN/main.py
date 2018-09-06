import tensorflow as tf
import model as ml
import data
import numpy as np

from configs import DEFINES

def main(self):
	inputTrainEncOneHot = []
	outputTrainDecOneHot = []
	# Make Vocabulary
	#######################################################################
	dictionary,  vocabularyLength = data.loadVocabulary()
	# print("#######################################################################")
	# print(dictionary)
	# print("#######################################################################")
	# print("#######################################################################")
	# print(vocabularyLength)
	# print("#######################################################################")
	######################################################################

	# Load Data
	#######################################################################
	xTrain, yTrain, xTest, yTest = data.loadData()
	# print("#######################################################################")
	# print(xTrain)
	# print("#######################################################################")
	# print("#######################################################################")
	# print(yTrain)
	# print("#######################################################################")
	######################################################################
	# index sequences 
	# pad sequences
	#print(len(xTrain))
	print("#######################################################################")
	#print(xTrain)
	#print(len(yTrain))
	print("#######################################################################")
	#print(yTrain)

	inputTrainEnc, inputTrainEncLength = data.encProcessing(xTrain, dictionary)
	outputTrainDec, outputTrainDecLength = data.decOutputProcessing(yTrain, dictionary)
	targetTrainDec = data.decTargetProcessing(yTrain, dictionary)
	#inputTrainEnc, inputTrainEncLength = data.decOutputProcessing(xTrain, dictionary)
	#outputTrainDec, outputTrainDecLength = data.encProcessing(yTrain, dictionary)
	# inputTestEnc, inputTestEncLength = data.encProcessing(xTest,dictionary)
	# outputTrainDec, outputTrainDecLength = data.decOutputProcessing(yTrain, dictionary)
	# outputTestDec, outputTestDecLength = data.decOutputProcessing(yTest, dictionary)
	# targetTrainDec = data.decTargetProcessing(yTrain, dictionary)
	# targetTestDec = data.decTargetProcessing(yTest, dictionary)

	#for encIndex in range(len(inputTrainEnc)):
	#	inputTrainEncOneHot.append(np.eye(vocabularyLength)[inputTrainEnc[encIndex]])
	
	#for decIndex in range(len(outputTrainDec)):
	#	outputTrainDecOneHot.append(np.eye(vocabularyLength)[outputTrainDec[decIndex]])

	# print(type(inputTrainEnc))
	# print(type(outputTrainDec))
	# print(type(targetTrainDec))
	print("#######################################################################")
	#print(np.asarray(inputTrainEncOneHot.shape))
	
	# inputTrainEnc = np.asarray(inputTrainEnc)
	# #print(inputTrainEnc)
		
	# outputTrainDec = np.asarray(outputTrainDec)
	# #print(outputTrainDec)
	# targetTrainDec = np.asarray(targetTrainDec)
	# #print(targetTrainDec)
	# print(targetTrainDec.shape)
	# # print("#######################################################################")
	# # print(targetTrainDec)
	# print("#######################################################################")
	# print("#######################################################################")
	# #print(outputTrainDec)
	# print(outputTrainDec.shape)
	# print("#######################################################################")
	# print("#######################################################################")
	# print(targetTrainDec.shape)
	#print(targetTrainDec)

	print("#######################################################################")	
	# print( type(inputTrainEnc))
	# print(outputTrainDec[0])
	# print(type(outputTrainDec))
	print(inputTrainEnc.shape)
	print(outputTrainDec.shape)
	print(targetTrainDec.shape)
	print("#######################################################################")	
	# inputTrainEnc = np.array(inputTrainEnc)
	# print( type(inputTrainEnc))
	# print(inputTrainEnc.shape)
	# outputTrainDec = np.array(outputTrainDec)
	# print( type(outputTrainDec))
	# print(outputTrainDec.shape)
	# print(outputTrainDec[0])
	# print(outputTrainDec[0].shape)
	# outputTrainDec[0] = outputTrainDec[0].reshape(1,15)
	# print( type(outputTrainDec))
	# print(outputTrainDec.shape)
	# print(outputTrainDec[0])
	# print(outputTrainDec[0].shape)
	print("#######################################################################")	
	######################################################################

	# classifier 
	#######################################################################
	classifier = tf.estimator.Estimator(
			model_fn=ml.Model, 
			params={
				'hiddenSize': DEFINES.hiddenSize,
				'layerSize': DEFINES.layerSize,
				'learningRate': DEFINES.learningRate,
				'vocabularyLength': vocabularyLength,
				'embeddingSize': DEFINES.embeddingSize,
			})
	print("#######################################################################")
	print(classifier)
	print("#######################################################################")	
	######################################################################

	# classifier xTrain 과 yTrain 들어 가는 부분에 먼가 차원이 들어 가야 하는 듯.
	#######################################################################
	classifier.train(input_fn=lambda:data.trainInputFn(inputTrainEnc, outputTrainDec, targetTrainDec,  DEFINES.batchSize), steps=DEFINES.trainSteps)
	#classifier.train(input_fn=lambda:data.trainInputFn(inputTrainEnc, outputTestDec, targetTrainDec,  DEFINES.batchSize), steps=DEFINES.trainSteps)
	#classifier.train(input_fn=lambda:data.trainInputFn(inputTrainEnc, outputTestDec, targetTrainDec, DEFINES.batchSize), steps=DEFINES.trainSteps)
	#######################################################################

	# evalResult = classifier.evaluate(
	# 		input_fn=lambda:data.evalInputFn(xTest, yTest, DEFINED.batchSize))

	# print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**evalResult))

#predictX 선언필요
#predictions = classifier.predict(
#			input_fn=lambda:data.evalInputFn(predictX,
#													labels=None,
#													batch_size=DEFINES.batchSize))
													
if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)
