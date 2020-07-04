#File containing activation functions

#Related third party imports
import numpy as np


def act_sigmoid(input):
	return (1/(1+np.exp(-input)))

def act_ReLU(input):
	
	return(np.maximum(0, input))

def act_straight(input):

	return input


def back_sigmoid(input):
	
	output = []

	for row in input:
		output.append(act_sigmoid(row) * (1 - act_sigmoid(row)))

	return np.squeeze(output, axis=1)


#Need to upadte...
def back_ReLU(input):
    input[input<=0] = 0
    input[input>0] = 1

    #print(input)
    #print(weights)

    return input



