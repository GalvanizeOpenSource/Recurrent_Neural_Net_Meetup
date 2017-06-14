# MLP 3 layer network: input - hidden - outer
# based on:
# http://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np

def activation(x,deriv=False):
    '''activation function, values for sigmoid'''
    if(deriv==True):
	    return x*(1-x) # derivative of sigmoid*
    return 1/(1+np.exp(-x))

# inputs (X)
inputs = np.array([[0,0],
                   [0,1],
                   [1,0],
                   [1,1]])

# targets (y) - presently for AND, change for OR and XOR
targets = np.array([[0],
			        [0],
			        [0],
			        [1]])


## make network geometry 
#nodes_input  = 
#nodes_hidden = 
#nodes_target = 

## initialize weights (-1.0 to 1.0)
#Wxh = 2*np.random.uniform(size=( , )) - 1
#Why = 2*np.random.uniform(size=( , )) - 1

## simulation parameters
#np.random.seed(1)
#alpha = 1 # learning rate
#num_epochs = 

# training pseudo code
# for each epoch: 
    # for each row of X, y in inputs, targets
        # Feed forward to find values of:
        # H
        # yp (the prediction)
        
        # Back propogate to find the gradient of the loss with respect to:
        # Why
        # Wxh

        # Use gradient descent to update the weights
    
    # for this epoch, print training error

# print final comparison between target and predictions
