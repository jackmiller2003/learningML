#File containing activation functions

#Related third party imports
import numpy as np

def act_sigmoid(input):
	return 1/(1+np.exp(-input))