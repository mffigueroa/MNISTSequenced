from keras.utils import Sequence

class SequenceCombiner(Sequence):
	def __init__(self, inputSequences, outputSequences):
		self.inputSequences = inputSequences
		self.outputSequences = outputSequences
		
		if len(inputSequences) < 1 or len(outputSequences) < 1:
			raise RuntimeError('Attempt to combine zero sequences.')
		
		allSequences = self.inputSequences + self.outputSequences
		
		commonLength = None
		for sequence in allSequences:
			sequenceLength = len(sequence)
			if commonLength is None:
				commonLength = sequenceLength
			elif sequenceLength != commonLength:
				raise RuntimeError('Attempt to combine sequences of unequal lengths.')
	
	def __len__(self):
		return len(self.inputSequences[0])
	
	def __getitem__(self, index):
		inputs = [ seq[index] for seq in self.inputSequences ]
		outputs = [ seq[index] for seq in self.outputSequences ]
		return inputs, outputs