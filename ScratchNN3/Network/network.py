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
					"\n\tActivations: \n" + str(self.activations) + "\n"
				)

	#Rewrite
	def forward(self, input_list):

		inputs = np.array(input_list, ndmin=2).T

		if self.layer_num == 1:

			self.activations = self.activation_function(inputs)

			return self.activations

		elif self.layer_num > 1:

			weighted_inputs = np.dot(np.matrix(self.weights), np.matrix(inputs).T) + np.transpose(np.matrix(self.bias), axes=None)

			self.weighted_inputs = weighted_inputs

			self.activations = self.activation_function(weighted_inputs)

			self.da_dz = self.back_function(self.weighted_inputs)

			return self.activations

		else:

			raise Exception("Iterated over a layer that doesn't exist")


	#Rewrite
	def update(self, lrate):

		self.weights = self.weights - lrate*self.delta

		self.bias = self.bias - lrate*self.bias_delta

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

		for layer in list(enumerate(self.layers)):
			
			if layer[0] == 0:

				layer[1].forward(input_array)

			else:

				layer[1].forward(self.layers[layer[0] - 1].activations)

		return self.layers[-1].activations


	#Rewrite
	def backward_prop(self, input_list, target_list):
		
		for layer in reversed(list(enumerate(self.layers))):

			#Last layer weights
			if layer[0] == len(list(enumerate(self.layers))) - 1:

				bias_delta = []

				delta = []
				neuron = 0

				while neuron < layer[1].neurons:

					sub_delta = []

					dc_da = cost_dMSE(input_list[neuron], target_list[neuron])
					da_dz = layer[1].da_dz[neuron]

					for weight in list(enumerate(layer[1].weights[neuron])):

						dz_dw = self.layers[layer[0] - 1].activations[weight[0]]
						sub_delta.append(dc_da * da_dz * dz_dw)

						#print(dc_da * da_dz * dz_dw)

					delta.append(np.squeeze(sub_delta))

					bias_delta.append(np.sum(np.squeeze(sub_delta)))

					neuron += 1

				layer[1].delta = np.squeeze(delta)

				layer[1].bias_delta = np.squeeze(bias_delta)


			#Hidden layer weights
			elif layer[0] != 0:

				bias_delta = []
				delta = []
				neuron = 0

				while neuron < layer[1].neurons:

					sub_delta = []
					da_dz = layer[1].da_dz[neuron]

					dc_da = np.sum(np.transpose(self.layers[layer[0] + 1].delta)[neuron]) / len(np.transpose(self.layers[layer[0] + 1].delta))

					for weight in list(enumerate(layer[1].weights[neuron])):

						dz_dw = self.layers[layer[0] - 1].activations[weight[0]]
						sub_delta.append(dc_da * da_dz * dz_dw)

					delta.append(np.squeeze(sub_delta))
					neuron += 1

					bias_delta.append(np.sum(np.squeeze(sub_delta)))

				layer[1].delta = np.squeeze(delta)

				layer[1].bias_delta = np.squeeze(bias_delta)


			else:
				continue


	def train(self, train_data, test_data, lrate, epochs):
		
		for epoch in range(epochs):

			for trdata in list(enumerate(train_data)):

				print("Back prop", trdata[0])

				prediction = self.forward(trdata[1][0])

				target = np.matrix(trdata[1][1]).T

				self.backward_prop(prediction, target)

				for layer in list(enumerate(self.layers)):

					if layer[0] != 0:

						#print(layer[1].delta)

						layer[1].update(lrate)

			score = 0

			for image in test_data:

				correct = max(enumerate(image[1]), key=operator.itemgetter(1))
				query = max(enumerate(n.forward(image[0])), key=operator.itemgetter(1))

				if correct[0] == query[0]:
					#print("success")
					score += 1
				else:
					#print("fail")
					continue

				
			print(score/len(test_data))

			
n = Network([
			{'layer_num': 1, 'category': 'Input', 'neurons': 784, 'activation_function': act_sigmoid},
			{'layer_num': 2, 'category': 'Hidden', 'neurons': 200, 'activation_function': act_sigmoid},
			{'layer_num': 3, 'category': 'Hidden', 'neurons': 25, 'activation_function': act_sigmoid},
			{'layer_num': 4, 'category': 'Output', 'neurons': 10, 'activation_function': act_sigmoid}
			])

#print(n.layers[2].bias)

#print(n.layers[1].backward(n.layers[2].activations))

data = dp.proc_data('Data/MNISTData/mnist_train.csv', 'Data/MNISTData/mnist_test.csv', 10)

n.train(data[0][:1000], data[1][:100], 0.1, 1)

#Code ain't working!
