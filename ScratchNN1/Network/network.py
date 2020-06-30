#Imports

#Standard library imports
import os, sys

#Related third party imports
import numpy as np
import matplotlib
import pprint as pp
import operator

#Local application/library specific imports
sys.path.append(os.path.join(os.path.dirname(__file__), "Data"))


from actfunc import *
from Data import dataproc as dp

class Network:


	def __init__(self, inodes, hnodes, onodes, lrate, activation):
		
		self.inodes = inodes
		self.hnodes = hnodes
		self.onodes = onodes
		
		self.lr = lrate

		self.wih = np.random.normal(
			float(0), pow(self.hnodes, -0.5), (self.hnodes, self.inodes)
			)

		self.who = np.random.normal(
			float(0), pow(self.onodes, -0.5), (self.onodes, self.hnodes)
			)

		self.act_func = activation

		pass

	def __str__(self):

		return ("\n" + "Network: \n" + " Inputs Nodes: " + str(self.inodes) + "\n Hidden Nodes: " + str(self.hnodes) +
			"\n Output Nodes: " + str(self.onodes) + "\n Learning Rate: " + str(self.lr) + "\n")


	def train(self, input_list, target_list):
		
		inputs = np.array(input_list, ndmin=2).T
		targets = np.array(target_list, ndmin=2).T

		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.act_func(hidden_inputs)

		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.act_func(final_inputs)	


		output_errors = targets - final_outputs
		hidden_errors = np.dot(self.who.T, output_errors)

		self.who += self.lr * np.dot(
			(output_errors * final_outputs * (float(1) - final_outputs)), np.transpose(hidden_outputs)
			)

		self.wih += self.lr * np.dot(
			(hidden_errors * hidden_outputs * (float(1) - hidden_outputs)), np.transpose(inputs)
			)

		pass

	def query(self, input_list):
		
		inputs = np.array(input_list, ndmin=2).T

		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.act_func(hidden_inputs)

		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.act_func(final_inputs)

		return final_outputs


n = Network(784, 25, 10, 0.2, act_sigmoid)

data = dp.proc_data('Data/MNISTData/mnist_train.csv', 'Data/MNISTData/mnist_test.csv', 10)

epochs = 1

for e in range(epochs):
	for image in data[0]:
		n.train(image[0], image[1])

score = 0

for image in data[1]:

	correct = max(enumerate(image[1]), key=operator.itemgetter(1))
	query = max(enumerate(n.query(image[0])), key=operator.itemgetter(1))

	#print(correct[0], query[0])

	if correct[0] == query[0]:
		#print("success")
		score += 1
	else:
		#print("fail")
		continue

print (score / len(data[1]))
