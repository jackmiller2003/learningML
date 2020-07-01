#Standard library imports
import os, sys

#Related third party imports
import numpy as np
import matplotlib.pyplot as plt
import pprint as pp
import operator

#Local application/library specific imports
sys.path.append(os.path.join(os.path.dirname(__file__), "Data"))
from actfunc import *
from Data import dataproc as dp

class Network:


	def __init__(self, structure, lrate):

		self.hlist = []

		for layer in range(len(structure)):

			if layer == 0:
				self.inodes = structure[layer][0]

			elif layer == (len(structure) - 1):
				self.onodes = structure[layer][0]
				self.oact = structure[layer][1]

			else:
				self.hlist.append(structure[layer])


		self.wih = np.random.normal(
			float(0), pow(self.hlist[0][0], -0.5), (self.hlist[0][0], self.inodes)
			)

		self.who = np.random.normal(
			float(0), pow(self.onodes, -0.5), (self.onodes, self.hlist[len(self.hlist)-1][0])
			)

		hlayer = 0
		self.hwlist = []

		while hlayer < (len(self.hlist)):
			if hlayer == (len(self.hlist) - 1):
				pass
			else:
				self.hwlist.append(np.random.normal(
					float(0), pow(self.hlist[hlayer][0], -0.5), (self.hlist[hlayer][0], self.hlist[hlayer + 1][0])
					))

			hlayer += 1

		#print(self.hwlist)

		self.lr = lrate

	def __str__(self):

		return ("\n" + "Network: \n" + " Inputs Nodes: " + str(self.inodes) + "\n Hidden Nodes: " + str(self.hlist) +
			"\n Output Nodes: " + str(self.onodes) + "\n Learning Rate: " + str(self.lr) + "\n")


	def train(self, input_list, target_list):
		
		inputs = np.array(input_list, ndmin=2)
		targets = np.array(target_list, ndmin=2).T

		self.query(inputs)

		output_error = targets - self.final_output

		hidden_error = np.dot(self.who.T, output_error)

		#Second term is the derivative of sigmoid
		self.who += self.lr * np.dot(
			(output_error * (self.final_output * (float(1) - self.final_output))), self.hlist[len(self.hlist) - 1][2].T
			)

		i = len(self.hwlist)
		#print(len(self.hwlist))

		while i > 0:
			
			n_hidden_error = np.dot(self.hwlist[i - 1], hidden_error)

			#Second term is the derivative of sigmoid
			self.hwlist[i - 1] += self.lr * (np.dot(
				(hidden_error * (self.hlist[i][2] * (float(1) - self.hlist[i][2]))), self.hlist[i - 1][2].T 
				)).T

			hidden_error = n_hidden_error

			i -= 1
		
		#Second term is the derivative of sigmoid
		self.wih += self.lr * np.dot(
			(hidden_error * (self.hlist[0][2] * (float(1) - self.hlist[0][2]))), inputs
			)

		pass

	def query(self, input_list):
		
		inputs = np.array(input_list, ndmin=2).T

		print(inputs)

		hidden_inputs = np.dot(self.wih, inputs)
		act_output = self.hlist[0][1](hidden_inputs)

		if len(self.hlist[0]) == 2:
			self.hlist[0].append(act_output)
		elif len(self.hlist[0]) == 3:
			self.hlist[0][2] = act_output

		hwlayer = 0
		while hwlayer < len(self.hwlist):

			weighted_inputs = np.dot(np.transpose(self.hwlist[hwlayer]), act_output)
			act_output = self.hlist[hwlayer + 1][1](weighted_inputs)
			
			if len(self.hlist[hwlayer + 1]) == 2:
				self.hlist[hwlayer + 1].append(act_output)

			elif len(self.hlist[hwlayer + 1]) == 3:
				self.hlist[hwlayer + 1][2] = act_output

			hwlayer += 1

		final_input = np.dot(self.who, act_output)
		final_output = self.oact(final_input)
		self.final_output = final_output

		return final_output

n = Network([[784],
[75, act_sigmoid],
[50 , act_sigmoid],
[10, act_sigmoid]], 0.3)

data = dp.proc_data('Data/MNISTData/mnist_train_100.csv', 'Data/MNISTData/mnist_test_10.csv', 10)

print(np.shape(data[0][0][0]))

print(np.shape(n.hwlist[0]))
print(n.hlist[0][0])

epochs = 1

for e in range(epochs):
	for image in data[0]:
		n.train(image[0], image[1])
		#print(n.hlist)'''

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
