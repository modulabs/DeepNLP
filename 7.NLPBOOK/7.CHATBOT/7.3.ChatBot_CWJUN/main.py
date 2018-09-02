import tensorflow as tf
import model as ml
import data

from configs import DEFINES

def main(self):

	# Make Vocabulary
	#######################################################################
	vocabularyDictionary,  vocabularyLength = data.loadVocabulary()
	print("#######################################################################")
	print(vocabularyDictionary)
	print("#######################################################################")
	print("#######################################################################")
	print(vocabularyLength)
	print("#######################################################################")
	######################################################################

	# Load Data
	#######################################################################
	xTrain, yTrain, xTest, yTest = data.loadData()
	print("#######################################################################")
	print(xTrain)
	print("#######################################################################")
	print("#######################################################################")
	print(yTrain)
	print("#######################################################################")
	######################################################################
	# index sequences 
	# pad sequences
	xTrain, yTrain, xTest, yTest = data.preProcessing(xTrain, yTrain, xTest, yTest)
	print("#######################################################################")
	print(xTrain)
	print("#######################################################################")
	print("#######################################################################")
	print(yTrain)
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
			})
	print("#######################################################################")
	print(classifier)
	print("#######################################################################")	
	######################################################################

	# classifier xTrain 과 yTrain 들어 가는 부분에 먼가 차원이 들어 가야 하는 듯.
	#######################################################################
	classifier.train(input_fn=lambda:data.trainInputFn(xTrain, yTrain, DEFINES.batchSize), steps=DEFINES.trainSteps)
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
