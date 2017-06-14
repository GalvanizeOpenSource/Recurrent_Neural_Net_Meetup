#Vanilla RNN predicts sine wave
#based on iamtrask's blog:
#http://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/
#and code in jakob aungiers github:
#https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction

import numpy as np
import matplotlib.pyplot as plt

#plotting
def plot_train_test(x, y, train_size):
    ''' for visualizing train-test split '''
    split = int(len(x)*train_size)
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]
    plt.plot(x_train, y_train, 'b-', label='train')
    plt.plot(x_test, y_test, 'r-', label='test')
    plt.legend()
    plt.show()

def plot_predict_vs_test(y_pred, y_test, figtitle):
    ''' visualize peformance on test set '''
    dps = np.arange(0,y_test.shape[0])
    plt.plot(dps, y_test, '-b', label='test')
    plt.plot(dps, y_pred, '-g', label='prediction')
    plt.legend()
    plt.title(figtitle)
    plt.show()

# activation function
def activation(x,deriv=False):
    '''activation function, values for tanh'''
    if(deriv==True):
	    return 1 - np.tanh(x)**2
    return np.tanh(x)

# inputs
def make_sine_wave(cycles, pts_per_cycle):
    ''' makes a sine wave, centered on 0, with peak-to-peak amplitude 
        of 2 (-1 to 1), with the desired number of cycles and the number of
        points per cycle
    '''
    x = np.linspace(0, cycles*2*np.pi, num=cycles*pts_per_cycle+1, endpoint=True)
    return x, np.sin(x)
    

def make_train_test_seqs(sequence, seq_len, train_size):
    ''' Takes a sequence and splits it into train - test groups based on
        the fractional train_size (0.8 means the train set is the first 80% of
        the data).  The seq_len is the desired number of datapoints in the 
        analysis window.
    '''
    seq_matrix = []
    sequence_length = seq_len + 1 # for now the target incl. in X as last value
    for index in range(len(sequence) - sequence_length):
        seq_matrix.append(sequence[index: index + sequence_length])
    seq_matrix = np.array(seq_matrix)
    last_row_train = int(round(train_size * seq_matrix.shape[0])) 
    train = seq_matrix[:last_row_train, :] # everything up to last_row_train is train
    np.random.shuffle(train) # shuffle these rows (but each row seq. preserved)
    X_train = train[:, :-1] # the last pt in each row not included in X_train
    y_train = train[:, -1] # the target was the last pt in each X_train row
    X_test = seq_matrix[last_row_train:, :-1] # same for test
    y_test = seq_matrix[last_row_train:, -1]

    return X_train, y_train, X_test, y_test

# inputs used to make the data
cycles = 50 # total number of sin wave cycles
pts_per_cycle = 50 # number of points per cycle
seq_len = 8  # number of datapoint in the training window
train_size = 0.8 # fraction of data set (from beginning) used to train

# simulation parameters
np.random.seed(1)
alpha = 0.1 # learning rate
num_epochs = 1000 # number of epochs

# make network geometry 
nodes_input  = seq_len # this is also the sequence training window
nodes_hidden = 16 
nodes_target = 1

# initialize weights 
Wxh = 2*np.random.uniform(size=(nodes_input, nodes_hidden)) - 1
Whh = 2*np.random.uniform(size=(nodes_hidden, nodes_hidden)) - 1
Why = 2*np.random.uniform(size=(nodes_hidden, nodes_target)) - 1

# make the data and visualize it
x, sinewave = make_sine_wave(cycles, pts_per_cycle)
plot_train_test(x, sinewave, train_size) # to check
# train - test split below
X_train, y_train, X_test, y_test = make_train_test_seqs(sinewave, seq_len, train_size)


# training
print("\nTraining:")
H_prev = np.zeros((1, nodes_hidden))
H_delta_fut = np.zeros(nodes_hidden)
for e in range(num_epochs):
    error_lst = [] # differences between target and predictions
    for X, y in zip(X_train, y_train):
        X = X.reshape((1, X.shape[0])) # for row, column shape
        y = y.reshape((1,1))
        # Feed forward 
        H = activation(np.dot(X,Wxh) + np.dot(H_prev,Whh))
        yp = activation(np.dot(H,Why))
        # Back propogate to find gradients
        # Why gradients 
        yp_error = y - yp
        yp_delta = yp_error*activation(yp,deriv=True)
        grad_Why = np.dot(H.T, yp_delta)
        # Wxh gradients 
        H_error = np.dot(yp_delta, Why.T) + np.dot(H_delta_fut, Whh.T)
        H_delta = H_error * activation(H,deriv=True)
        #H_delta_fut = np.copy(H_delta) crashes simulation 
        grad_Wxh = np.dot(X.T, H_delta)
        # Whh gradients
        grad_Whh = np.dot(H_prev.T, H_delta)
        # Use gradient descent to update weights
        Why += alpha * grad_Why
        Whh += alpha * grad_Whh
        Wxh += alpha * grad_Wxh
        # save for future use
        H_prev = np.copy(H)
        error_lst.append(np.abs(yp_error[0][0]))
    
    epoch_error = np.mean(error_lst).round(4)
    if (e % int(num_epochs/20)) == 0:
        print("Epoch: {0:<8s} Error: {1}".format(str(e), epoch_error))

print("Simulation finished.")
print("Testing.")

# Test on test set - just predict 1 ahead
yp_lst = [] # predictions
H_prev = np.zeros((1, nodes_hidden))
for X in X_test:
    X = X.reshape((1, X.shape[0])) # for row, column shape
    # Feed forward 
    H = activation(np.dot(X,Wxh) + np.dot(H_prev,Whh))
    yp = activation(np.dot(H,Why))
    yp_lst.append(yp[0][0])
    H_prev = np.copy(H)

plot_predict_vs_test(yp_lst, y_test, 'Test - only plotting 1 ahead')

# Test on test set - after initial seed predict all the rest
yp_lst = [] # predictions
H_prev = np.zeros((1, nodes_hidden))
for i in range(X_test.shape[0]):
    if i == 1:
        X = X_test[0]
        X = X.reshape((1, X.shape[0]))
    # Feed forward 
    H = activation(np.dot(X,Wxh) + np.dot(H_prev,Whh))
    yp = activation(np.dot(H,Why))
    yp_lst.append(yp[0][0])
    H_prev = np.copy(H)
    X = np.append(X[:,1:], yp[0]).reshape((1, nodes_input))

plot_predict_vs_test(yp_lst, y_test, 'Test - predicting entire test after seed')

