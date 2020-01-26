import numpy as np
import enum
from keras.utils import Sequence

class SequenceToSequenceDataType(enum.Enum):
	TargetData = 1
	InputData = 2

class SequenceToSequenceData(Sequence):
	def __init__(self, sequenceData, outputDataType, startSymbol):
		self.sequenceData = sequenceData
		self.outputDataType = outputDataType
		self.startSymbol = startSymbol
	
	def __len__(self):
		return len(self.sequenceData)
	
	def __getitem__(self, index):
		sequenceBatch = self.sequenceData[index]
		batchSize = sequenceBatch.shape[0]
		
		if self.outputDataType == SequenceToSequenceDataType.InputData:
			startSymbolAsSingleBatch = self.startSymbol[np.newaxis, np.newaxis, :]
			startSymbolBatched = np.repeat(startSymbolAsSingleBatch, batchSize, axis=0)
			sequenceBatchWithoutTerminator = sequenceBatch[:, :-1, :]
			sequenceInput = np.concatenate([startSymbolBatched, sequenceBatchWithoutTerminator], axis=1)
			return sequenceInput
		else:
			return sequenceBatch