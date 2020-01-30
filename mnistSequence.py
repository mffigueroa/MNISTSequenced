import os
import numpy as np
from keras.utils import Sequence
import pickle
from shuffledList import ShuffledList

class MNISTSequence(Sequence):
	def __init__(self, dataDirectory, batchSize, isTrain=True):
		pickleFilePath = os.path.join(dataDirectory, 'MNISTSequence.pickle')
		loadedPickleFile = False
		try:
			with open(pickleFilePath, 'rb') as pickleFile:
				pickleObj = pickle.load(pickleFile)
				self.trainingFiles = pickleObj['trainingFiles']
				self.testFiles = pickleObj['testFiles']
				if isTrain:
					self.sequences = pickleObj['trainSequences']
				else:
					self.sequences = pickleObj['testSequences']
				loadedPickleFile = True
		except:
			pass
				
		if not loadedPickleFile:
			self.trainingFiles, self.testFiles = self.GetSequenceFiles(dataDirectory)
			trainSequences = []
			testSequences = []
			for filePath in self.trainingFiles:
				trainSequences.append(self.ParseSequenceFile(filePath))
			for filePath in self.testFiles:
				testSequences.append(self.ParseSequenceFile(filePath))
			pickleObj = { 'trainingFiles' : self.trainingFiles,
				'testFiles' : self.testFiles,
				'trainSequences' : trainSequences,
				'testSequences' : testSequences }
			with open(pickleFilePath, 'wb') as pickleFile:
				pickle.dump(pickleObj, pickleFile)
			
			if isTrain:
				self.sequences = ShuffledList(trainSequences)
			else:
				self.sequences = ShuffledList(testSequences)
		
		sequenceLengths = [ sequence.shape[0] for sequence in self.sequences ]
		self.maxSequenceLength = max(sequenceLengths)
		self.sequenceElementSize = self.sequences[0].shape[1]
		self.batchSize = batchSize
		self.numBatches = len(self.sequences) // self.batchSize
	
	def __len__(self):
		return self.numBatches
	
	def __getitem__(self, index):
		batch = np.zeros((self.batchSize,
			self.maxSequenceLength, self.sequenceElementSize))
		batchStartOffset = index * self.batchSize
		for batchIndex in range(self.batchSize):
			index = batchStartOffset + batchIndex
			sequence = self.sequences[index]
			batch[batchIndex, :sequence.shape[0], :] = sequence
		return batch
	
	def on_epoch_end(self):
		self.sequences.shuffle()
	
	def GetSequenceFiles(self, directory):
		sequenceFiles = os.listdir(directory)
		
		trainingFiles = []
		testFiles = []
		
		trainFilenamePrefix = 'trainimg-'
		testFilenamePrefix = 'testimg-'
		trainPrefixLen = len(trainFilenamePrefix)
		testPrefixLen = len(testFilenamePrefix)
		for filename in sequenceFiles:
			if not 'inputdata' in filename:
				continue
			fullFilepath = os.path.join(directory, filename)
			if len(filename) >= trainPrefixLen and filename[:trainPrefixLen] == trainFilenamePrefix:
				trainingFiles.append(fullFilepath)
			elif len(filename) >= testPrefixLen and filename[:testPrefixLen] == testFilenamePrefix:
				testFiles.append(fullFilepath)
		
		return trainingFiles, testFiles
	
	def ParseSequenceFile(self, filePath):
		return np.loadtxt(filePath)