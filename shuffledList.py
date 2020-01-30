import numpy as np

class ShuffledList(list):
	def __init__(self):
		self.shuffledIndices = np.arange(len(self))
	
	def __getitem__(self, index):
		shuffledIndex = self.shuffledIndices[index]
		return super(ShuffledList, self).__getitem__(shuffledIndex)
	
	def __setitem__(self, index, value):
		shuffledIndex = self.shuffledIndices[index]
		return super(ShuffledList, self).__setitem__(shuffledIndex, value)
	
	def __delitem__(self, index):
		shuffledIndex = self.shuffledIndices[index]
		super(ShuffledList, self).__delitem__(shuffledIndex)
		shuffledIndices = list(self.shuffledIndices)
		del shuffledIndices[index]
		self.shuffledIndices = np.array(shuffledIndices)
	
	def shuffle(self):
		np.random.shuffle(self.shuffledIndices)