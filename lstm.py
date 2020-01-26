import keras
import numpy as np

from keras.models import Model
from keras.layers import LSTM, Input, Dense, Lambda, TimeDistributed, Concatenate

class LSTMModel(object):
	def __init__(self, hiddenUnits, numLayers, inputShape, maxDeltaMag=28):
		self.sequenceShape = (None,) + inputShape
		self.maxDeltaMag = maxDeltaMag
		self.numLayers = numLayers
		self.hiddenUnits = hiddenUnits
		self.encoder_input = Input(shape=self.sequenceShape)
		
		x = self.encoder_input
		
		self.encoder_hiddenStates = []
		self.encoder_cellStates = []
		for layerNum in range(self.numLayers):
			x, encoder_hidden, encoder_cell = LSTM(self.hiddenUnits, return_state=True)(x)
			self.encoder_hiddenStates.append(encoder_hidden)
			self.encoder_cellStates.append(encoder_cell)
		
		self.decoder_input = Input(shape=self.sequenceShape)
		x = self.decoder_input
		
		self.decoder_lstm_layers = []
		for layerNum in range(self.numLayers):
			lstm = LSTM(self.hiddenUnits, return_state=True, return_sequences=True)
			self.decoder_lstm_layers.append(lstm)
	
	def getFinalOutputLayers(self, lstmOutput):
		maxDeltaMagCopy = self.maxDeltaMag
		dxy = TimeDistributed(Dense(2, activation='tanh'))(lstmOutput)
		dxy = TimeDistributed(Lambda(lambda x: maxDeltaMagCopy*(2*x - 1)))(dxy)
		endOfStroke = TimeDistributed(Dense(1, activation='tanh'))(lstmOutput)
		endOfDigit = TimeDistributed(Dense(1, activation='tanh'))(lstmOutput)
		return Concatenate(axis=-1)([dxy, endOfStroke, endOfDigit])
	
	def buildTrainingModel(self):
		x = self.decoder_input
		
		for layerNum in range(self.numLayers):
			decoder_lstm_layer = self.decoder_lstm_layers[layerNum]
			encoder_hiddenState = self.encoder_hiddenStates[layerNum]
			encoder_cellState = self.encoder_cellStates[layerNum]
			x, _, _ = decoder_lstm_layer(x, initial_state=[encoder_hiddenState, encoder_cellState])
		
		finalOutput = self.getFinalOutputLayers(x)
		encoder_decoder_model = Model(inputs=[self.encoder_input, self.decoder_input], outputs=[finalOutput])
		return encoder_decoder_model
	
	def buildPredictionModel(self):
		total_encoder_states = self.numLayers * 2
		
		decoder_state_inputs = [Input(shape=(self.hiddenUnits,)) for i in range(total_encoder_states)]
		x = self.decoder_input
		decoder_inputs = [self.decoder_input] + decoder_state_inputs
		decoder_hiddenStates = []
		decoder_cellStates = []
		
		for layerNum in range(self.numLayers):
			encoder_hiddenState = decoder_state_inputs[layerNum]
			encoder_cellState = decoder_state_inputs[layerNum + self.numLayers]
			decoder_lstm_layer = self.decoder_lstm_layers[layerNum]
			layer_initial_state = [encoder_hiddenState, encoder_cellState]
			x, decoder_hiddenState, decoder_cellState = decoder_lstm_layer(x, initial_state=layer_initial_state)
			decoder_hiddenStates.append(decoder_hiddenState)
			decoder_cellStates.append(decoder_cellState)
		
		finalOutput = self.getFinalOutputLayers(x)
		decoder_outputs = [finalOutput] + decoder_hiddenStates + decoder_cellStates
		decoder_model = Model(inputs=decoder_inputs, outputs=decoder_outputs)		
		
		encoder_model_outputs = self.encoder_hiddenStates+self.encoder_cellStates
		encoder_model = Model(inputs=[self.encoder_input], outputs=encoder_model_outputs)
		return encoder_model, decoder_model