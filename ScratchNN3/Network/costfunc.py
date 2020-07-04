#File containing cost functions

#Related third party imports
import numpy as np

#https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
def cost_bincrsentr(predictions, targets):

    cost = -1 / (predictions.shape[1]) * (np.dot(targets, np.log(predictions).T) + np.dot(1 - targets, np.log(1 - predictions).T))
    return np.squeeze(cost)


#https://stackoverflow.com/questions/47377222/what-is-the-problem-with-my-implementation-of-the-cross-entropy-function
def cost_crsentr(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

def cost_MSE(predictions, targets):

    return np.square(np.subtract(targets,predictions))

def cost_dMSE(predictions, targets):

    return (2*(np.subtract(targets, predictions)))