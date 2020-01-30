from functools import reduce
from operator import add
from keras.utils import Sequence

class SequenceSubset(Sequence):
	# subsets should be a list of percentage values that sum up to 1
	# and describe the way the data should be partitioned.
	# subsetIndex should then be an index into that set that specifies
	# which of the partitions this object should retrieve its batches from.
	def __init__(self, sequenceData, subsets, subsetIndex):
		if subsetIndex < 0 or subsetIndex >= len(subsets):
			raise IndexError('subsetIndex out of range.')
		totalPercentage = reduce(add, subsets)
		if totalPercentage != 1.0:
			raise ValueError('Percentages in subsets should sum to 1')
		self.sequenceData = sequenceData
		fullDataLength = len(sequenceData)
		self.subsetOffset = reduce(add, subsets[:subsetIndex], 0)
		self.subsetOffset = int(self.subsetOffset * fullDataLength)
		self.subsetLength = int(subsets[subsetIndex] * fullDataLength)
		
	def __len__(self):
		return self.subsetLength
	
	def __getitem__(self, index):
		index += self.subsetOffset
		return self.sequenceData[index]