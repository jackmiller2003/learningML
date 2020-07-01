#File containing activation functions

#Related third party imports
import numpy as np


def act_sigmoid(input):
	return (1/(1+np.exp(-input)))

def act_ReLU(input):
	
	return(np.maximum(0, input))


def back_sigmoid(dA, input):
	
	return dA * act_sigmoid(input) * (1 - act_sigmoid(input))

def back_ReLU(dA, input):

	dZ = np.array(dA, copy=True)

	dZ[input <= 0] = 0
	return dZ

