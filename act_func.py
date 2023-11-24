import numpy as np


def step_function(x):
    """
    Return step function output
    return=1 if >0
    return=0 if x<=0
    
    Parameter
    ---------
    x : numpy array
    
    Return
    ------
    result: numpy array
      - consists of 0 or 1
    
    """
    y = x>0
    return y.astype(np.int64)

def sigmoid(x):
    """
    return sigmoid output
    """
    result = 1/(1+np.exp(-x))
    return result    
    
def relu(x):
    """
    return ReLU output
    """
    result = np.maximum(0,x)
    return result

def softmax(x):
    c=np.max(x)
    exp_x = np.exp(x-c) # overflow 대책
    exp_sum = np.sum(exp_x)
    result = exp_x/exp_sum
    return result