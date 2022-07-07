import numpy as np
from matplotlib import pyplot as plt

# AND Gate 
# possible combination of inputs
features = np.array(
    [
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1]
    ])

# target output
target = np.array([-1, 1, 1, 1])
  
  
# initialise weights, bias , learning rate, epoch
weight = [0.2, 0.3] # w1 w2
w = np.array([0.1, 0.2, 0.3])
bias = 1 # b
learning_rate = 0.01 # alpha
epoch = 1 # number of times the entire dataset is seen
square_error = []

for i in range(epoch):
    
    sum_squared_error = 0.0
    
    # for each of the possible input given in the features
    print(w)
    print(w.transpose())