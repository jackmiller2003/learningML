#Imports

#Standard library imports
import os, sys

#Related third party imports
import numpy as np
import matplotlib.pyplot as plt
import pprint as pp

#Local application/library specific imports
sys.path.append(os.path.join(os.path.dirname(__file__), "MNISTData"))

def proc_data(train_data, test_data, onodes):

	train_data = str(train_data)
	test_data = str(test_data)

	trdata_file = open(train_data, "r")
	trdata_list = trdata_file.readlines()
	print("Len. of traing data: " + str(len(trdata_list)))
	trdata_file.close()

	tedata_file = open(test_data, "r")
	tedata_list = tedata_file.readlines()
	print("Len. of test data: " + str(len(tedata_list)))
	tedata_file.close()

	def inp_tar(data):
		final_list = []
		for record in data:
			values = record.split(',')
			inputs = ((np.asfarray(values[1:]) / float(255)) * 0.99) + 0.01
			targets = np.zeros(onodes) + 0.01
			targets[int(values[0])] = 0.99
			final_list.append([inputs, targets])
		return final_list


	return inp_tar(trdata_list), inp_tar(tedata_list)

print("yes")

#result = proc_data('MNISTData/mnist_train.csv', 'MNISTData/mnist_test.csv', 10)


