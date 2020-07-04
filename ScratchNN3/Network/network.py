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
from Data import dataproc as dp
sys.path.append(os.path.join(os.path.dirname(__file__), "Data"))

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

			self.bias = np.random.normal(float(0), pow(3, -0.5), (neurons))


	def __str__(self):

		if self.activations is None:

			return ("\n" + "Layer " + str(self.layer_num) + "\n\tNeurons: " + str(self.neurons) +
					"\n\tActivation: " + str(self.activation_function) + "\n\tCategory: " + str(self.category) + "\n" +
					"\tNote: No activations set!" + "\n"
				)

		else:

			return ("\n" + "Layer " + str(self.layer_num) + "\n\tNeurons: " + str(self.neurons) +
					"\n\tActivation Function: " + str(self.activation_function) + "\n\tCategory: " + str(self.category) +
					"\n\tActivations: \n" + str(self.activations) + "\n"
				)

	#Rewrite
	def forward(self, input_list):

		inputs = np.array(input_list, ndmin=2).T

		if self.layer_num == 1:

			self.activations = self.activation_function(inputs)

			return self.activations

		elif self.layer_num > 1:

			z = np.dot(np.matrix(self.weights), np.matrix(inputs).T) + np.transpose(np.matrix(self.bias), axes=None)

			self.z = z

			self.activations = self.activation_function(z)

			return self.activations

		else:

			raise Exception("Iterated over a layer that doesn't exist")


	#Rewrite
	def update(self, lrate):

		pass

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

		for layer in list(enumerate(self.layers)):
			
			if layer[0] == 0:

				layer[1].forward(input_array)

			else:

				#print(layer[0] - 1)

				#print(self.layers[layer[0] - 1].activations)

				print(str(layer[0] - 1), layer[1].forward(self.layers[layer[0] - 1].activations))

				layer[1].forward(self.layers[layer[0] - 1].activations)

		return self.layers[-1].activations


	#Rewrite
	def backward_prop(self, input_list, target_list, lrate):

		#print(type(input_list), input_list)

		#print(type(target_list), target_list)
		
		prediction = self.forward(input_list)

		dc_da = cost_dMSE(prediction, np.transpose(np.matrix(target_list)))

		dSigmaZ = self.layers[-1].back_function(self.layers[-1].z)

		self.layers[-1].loss = np.multiply(dc_da, dSigmaZ)

		#print(self.layers[-1].output_loss)

		for hlayer in reversed(list(enumerate(self.layers[1:3]))):

			#print(hlayer[1])

			dSigmaZ = hlayer[1].back_function(hlayer[1].z)

			#print(self.layers[hlayer[0] + 2])

			hlayer[1].loss = np.multiply(
				(np.transpose(self.layers[hlayer[0] + 2].weights) * self.layers[hlayer[0] + 2].loss),
				dSigmaZ
				)
			
			#print(hlayer[1].loss)

		for layer in list(enumerate(self.layers[1:])):
			
			#print(np.transpose(np.matrix(layer[1].bias)) - layer[1].loss)
			
			layer[1].bias = np.transpose(np.matrix(layer[1].bias)) - lrate*(layer[1].loss)

			#print(layer[1].weights)

			#print(layer[1].weights - lrate*(np.dot(layer[1].loss, np.transpose(self.layers[layer[0]].activations))))

			layer[1].weights = layer[1].weights - lrate*(np.dot(layer[1].loss, np.transpose(self.layers[layer[0]].activations)))


	def train(self, train_data, test_data, lrate, epochs):
		
		for epoch in range(epochs):

			for data in train_data:

				print(data)

				input_list = data[0]
				output = data[1]

				#print(type(input_list), input_list)
				#print(type(output), output)

				self.backward_prop(input_list, output, lrate)


			for data in test_data:

				pass

			#self.forward(data[0])

			#print(forward(data[1]))
		
		
			
n = Network([
			{'layer_num': 1, 'category': 'Input', 'neurons': 5, 'activation_function': act_sigmoid},
			{'layer_num': 2, 'category': 'Hidden', 'neurons': 3, 'activation_function': act_sigmoid},
			{'layer_num': 3, 'category': 'Hidden', 'neurons': 3, 'activation_function': act_sigmoid},
			{'layer_num': 4, 'category': 'Output', 'neurons': 5, 'activation_function': act_sigmoid}
			])

#print(n.layers[2].bias)

#print(n.layers[1].backward(n.layers[2].activations))

#n.backward_prop([0.1, 0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.1, 0.1 ], 0.1)

n.train([[[0.1, 0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.1, 0.1 ]]], 
	[[[0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1]]],
	0.1, 1)

#n.forward([0.1, 0.1, 0.2, 0.3, 0.4])

'''n.train([[[0.1, 0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.1, 0.1 ]]], 
	[[[0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1]]],
	0.1, 2)'''

#Code ain't working!
