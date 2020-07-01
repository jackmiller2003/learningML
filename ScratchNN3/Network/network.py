#Standard library imports
import os, sys

#Related third party imports
import numpy as np
import matplotlib.pyplot as plt
import pprint as pp
import operator

#Local application/library specific imports
from actfunc import *
from costfunc import *

class LayerDense:

	def __init__(self, layer_num, neurons, prev_neurons, activation_function, category):

		self.layer_num = layer_num
		self.neurons = neurons

		self.activation_function = activation_function
		
		if self.activation_function == act_sigmoid:
			self.back_function = back_sigmoid

		elif self.activation_function == act_ReLU:
			self.back_function = back_ReLU

		else:
			raise Exception('No backward function was defined')

		self.category = category
		self.activations = None

		if layer_num != 1:

			self.weights = np.random.normal(
				float(0), pow(neurons, -0.5), (neurons, prev_neurons)
				)

			self.bias = np.random.normal(float(0), pow(3, -0.5), neurons)

	def __str__(self):

		if self.activations is None:

			return ("\n" + "Layer " + str(self.layer_num) + "\n\tNeurons: " + str(self.neurons) +
					"\n\tActivation: " + str(self.activation_function) + "\n\tCategory: " + str(self.category) + "\n" +
					"\tNote: No activations set!" + "\n"
				)

		else:

			return ("\n" + "Layer " + str(self.layer_num) + "\n\tNeurons: " + str(self.neurons) +
					"\n\tActivation Function: " + str(self.activation_function) + "\n\tCategory: " + str(self.category) +
					"\n\tActivations: " + str(self.activations) + "\n"
				)

	def forward(self, input_list):

		inputs = np.array(input_list, ndmin=2).T

		if self.layer_num == 1:

			self.activations = self.activation_function(inputs)

			return self.activation_function(inputs)

		else:

			weighted_inputs = np.dot(np.matrix(self.weights), np.matrix(inputs).T) + np.transpose(np.matrix(self.bias), axes=None)

			self.weighted_inputs = weighted_inputs

			self.activations = self.activation_function(weighted_inputs)

			return self.activation_function(weighted_inputs)

	#https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
	def backward(self, prev_act):
		
		scale = prev_act.shape[1]

		dBackAct = self.back_function(self.activations, self.weighted_inputs)

		dBackWeights = np.dot(dBackAct, prev_act.T) / scale

		dBackBiases = np.sum(dBackAct, axis=1, keepdims=True) / scale

		dPrevAct = np.dot(self.weights.T, dBackAct)

		return dPrevAct, dBackWeights, dBackBiases	

class Network:

	def __init__(self, structure):

		self.layers = []

		for layer in structure:
			if layer['layer_num'] == 1:
				self.layers.append(LayerDense(
					layer_num=layer['layer_num'], neurons=layer['neurons'], prev_neurons=0, 
					activation_function=layer['activation_function'], category=layer['category']
					))
				iprev_neurons = layer['neurons']
			else:
				self.layers.append(LayerDense(
					layer_num=layer['layer_num'], neurons=layer['neurons'], prev_neurons=iprev_neurons, 
					activation_function=layer['activation_function'], category=layer['category']
					))
				iprev_neurons = layer['neurons']

	def forward(self, input_list):

		input_array = np.array(input_list, ndmin=2)

		#print(input_array)

		for layer in enumerate(self.layers):
			
			if layer[0] == 0:

				layer[1].forward(input_array)

			else:

				layer[1].forward(self.layers[layer[0] - 1].activations)

		return self.layers[-1].activations

	def backward(self, target_list):
		pass

n = Network([
			{'layer_num': 1, 'category': 'Input', 'neurons': 10, 'activation_function': act_sigmoid},
			{'layer_num': 2, 'category': 'Hidden', 'neurons': 5, 'activation_function': act_ReLU},
			{'layer_num': 2, 'category': 'Hidden', 'neurons': 6, 'activation_function': act_ReLU},
			{'layer_num': 2, 'category': 'Output', 'neurons': 10, 'activation_function': act_sigmoid}
			])

#print(n.layers[2].bias)

n.forward([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print(n.layers[1].backward(n.layers[2].activations))
