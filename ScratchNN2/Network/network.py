#Standard library imports
import os, sys

#Related third party imports
import numpy as np
import matplotlib
import pprint as pp
import operator

#Local imports
from actfunc import *

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
		pass


	def query(self, input_list):
		
		inputs = np.array(input_list, ndmin=2).T

		hidden_inputs = np.dot(self.wih, inputs)
		hidden_output1 = self.hlist[0][1](hidden_inputs)
		print(hidden_output1)

		hwlayer = 0
		while hwlayer < len(self.hwlist):
			if hwlayer == 0:
				#print(self.hwlist[hwlayer])
				weighted_inputs = np.dot(np.transpose(self.hwlist[hwlayer]), hidden_output1)
				act_output = self.hlist[hwlayer + 1][1](weighted_inputs)
				print(act_output)
				hwlayer += 1
			else:
				#print(self.hwlist[hwlayer - 1])
				weighted_inputs = np.dot(np.transpose(self.hwlist[hwlayer]), act_output)
				act_output = self.hlist[hwlayer + 1][1](weighted_inputs)
				print(act_output)
				hwlayer += 1

		final_inputs = np.dot(self.who, act_output)
		final_outputs = self.oact(final_inputs)

		return final_outputs

		#final_inputs = np.dot(self.who, hidden_outputs)
		#final_outputs = self.act_func(final_inputs)

		#return final_outputs




n = Network([[5],
[5, act_sigmoid],
[6, act_sigmoid],
[7, act_sigmoid], 
[2, act_sigmoid]], 1)

print(n.query([1, 1, 1, 1, 1]))