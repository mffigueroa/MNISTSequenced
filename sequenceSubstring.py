import numpy as np
from keras.utils import Sequence

class SequenceSubstring(Sequence):
	def __init__(self, sequenceData, seqSubstringOffset=0, seqSubstringLen=0):
		self.sequenceData = sequenceData
		self.seqSubstringOffset = seqSubstringOffset
		self.seqSubstringLen = seqSubstringLen
	
	def __len__(self):
		return len(self.sequenceData)
		
	def __getitem__(self, index):
		batch = self.sequenceData[index]		
		batch = batch[:, self.seqSubstringOffset:, :]
		if self.seqSubstringLen > 0:
			batch = batch[:, :self.seqSubstringLen, :]
		return batch