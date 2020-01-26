import numpy as np
from sequenceToSequenceData import SequenceToSequenceData, SequenceToSequenceDataType
from mnistSequence import MNISTSequence
from lstm import LSTMModel
from sequenceCombiner import SequenceCombiner

import code

decoderStartSymbol = np.zeros((4,))

encoderInput = MNISTSequence(r'data\sequences', 8, seqSubstringLen=5)
decoderTarget = MNISTSequence(r'data\sequences', 8, seqSubstringOffset=5)
decoderInput = SequenceToSequenceData(decoderTarget, SequenceToSequenceDataType.InputData, decoderStartSymbol)
combinedGenerator = SequenceCombiner([encoderInput, decoderInput], [decoderTarget])

model = LSTMModel(16, 1, (4,))
trainingModel = model.buildTrainingModel()
trainingModel.compile(optimizer='Adam', loss='mse')
trainingModel.fit_generator(combinedGenerator, epochs=1000)