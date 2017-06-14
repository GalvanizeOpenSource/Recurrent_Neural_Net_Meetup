# MLP 3 layer network: input - hidden - outer
# based on:
# http://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np
import pdb

def activation(x,deriv=False):
    '''activation function, values for sigmoid'''
    if(deriv==True):
	    return x*(1-x) # not technically correct
    return 1/(1+np.exp(-x))

# inputs
inputs = np.array([[0,0],
                   [0,1],
                   [1,0],
                   [1,1]])
# targets                 
targets = np.array([[0],
			        [1],
			        [1],
			        [0]])


# make network geometry 
nodes_input  = 2
nodes_hidden = 4
nodes_target = 1

# initialize weights 
Wxh = 2*np.random.uniform(size=(nodes_input, nodes_hidden)) - 1
Why = 2*np.random.uniform(size=(nodes_hidden, nodes_target)) - 1

# simulation parameters
np.random.seed(1)
alpha = 1 # learning rate
num_epochs = 20000 # number of epochs

# training
print("\nTraining:")
for e in range(num_epochs):
    yp_lst = [] # predictions
    error_lst = [] # differences between target and predictions
    for X, y in zip(inputs, targets):
        X = X.reshape((1, X.shape[0])) # for row, column shape
        # Feed forward 
        H = activation(np.dot(X,Wxh))
        yp = activation(np.dot(H,Why))
        # Back propogate to find gradients
        # Why gradients 
        yp_error = y - yp
        yp_delta = yp_error*activation(yp,deriv=True)
        grad_Why = np.dot(H.T, yp_delta)
        # Wxh gradients 
        H_error = np.dot(yp_delta, Why.T)
        H_delta = H_error * activation(H,deriv=True)
        grad_Wxh = np.dot(X.T, H_delta)
        # Use gradient descent to update weights
        Why += alpha * grad_Why
        Wxh += alpha * grad_Wxh
        
        # save for future use
        yp_lst.append(yp[0][0])
        error_lst.append(yp_error[0][0])
    
    epoch_error = np.mean(np.abs(error_lst)).round(4)
    if (e % int(num_epochs/10)) == 0:
        print("Epoch: {0:<8s} Error: {1}".format(str(e), epoch_error))

print("\nResults:") 
print("yt\typ\typ_prob")
for yt, yp in zip(targets,yp_lst):
    print("{0}\t{1}\t{2}".format(yt[0], int(yp>=0.5), round(yp,4)))
