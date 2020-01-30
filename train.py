import numpy as np
from sequenceToSequenceData import SequenceToSequenceData, SequenceToSequenceDataType
from mnistSequence import MNISTSequence
from lstm import LSTMModel
from sequenceCombiner import SequenceCombiner
from sequenceSubset import SequenceSubset
from sequenceSubstring import SequenceSubstring
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

import code

def CosineAnnealingSchedule(lr_min, lr_max, periods_until_warm_restart):
	def scheduleFunction(epoch):
		lr = lr_min
		lr += (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / periods_until_warm_restart)) / 2
		return lr
	return LearningRateScheduler(scheduleFunction)

mnist_trainData = MNISTSequence(r'data\sequences', 8)

encoderInput = SequenceSubstring(mnist_trainData, seqSubstringLen=5)
decoderTarget = SequenceSubstring(mnist_trainData, seqSubstringOffset=5)
decoderStartSymbol = np.zeros((4,))
decoderInput = SequenceToSequenceData(decoderTarget, SequenceToSequenceDataType.InputData, decoderStartSymbol)
trainData_inputsOutputs = SequenceCombiner([encoderInput, decoderInput], [decoderTarget])

mnist_trainData_subsets = [ 0.85, 0.15 ]
mnist_trainSubset = SequenceSubset(trainData_inputsOutputs, mnist_trainData_subsets, 0)
mnist_validationSubset = SequenceSubset(trainData_inputsOutputs, mnist_trainData_subsets, 1)

lstmCells = 16
lstmLayers = 1
modelCheckPointPrefix = 'lstmModel_{}units_{}layer'.format(lstmCells, lstmLayers)

model = LSTMModel(lstmCells, lstmLayers, (4,))
trainingModel = model.buildTrainingModel()
trainingModel.compile(optimizer='Adam', loss='mse')
trainingModel.fit_generator(mnist_trainSubset,
	epochs=1000, validation_data=mnist_validationSubset,
	callbacks=[ ModelCheckpoint(modelCheckPointPrefix, period=1),
	CosineAnnealingSchedule(1e-5, 1e-2, 2)
	])