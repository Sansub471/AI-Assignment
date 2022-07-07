import numpy as np
from matplotlib import pyplot as plt

# AND Gate 
# possible combinations
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
bias = 1 # b
alpha = 0.001 # alpha
epoch = 1000
square_error = []

for i in range(epoch):
    
    sum_squared_error = 0.0
    
    # for each of the possible input given in the features
    for j in range(features.shape[0]):
  
        # actual output to be obtained
        actual = target[j]
  
        # the value of two features as given in the features array
        x1 = features[j][0]
        x2 = features[j][1]
  
        # Yin
        unit = (x1 * weight[0]) + (x2 * weight[1]) + bias
        
        # ( t - Yin)
        error = actual - unit
        delW1 = x1 * alpha * error
        delW2 = x2 * alpha * error
        deltaW = max(delW1, delW2)

        # summation of squared error is calculated
        # sum_squared_error += error * error
  
        # update weights
        weight[0] += alpha * error * x1
        weight[1] += alpha * error * x2
        

        # update bias       
        bias += alpha * error

        if deltaW < 0.003:
            break
    square_error.append(deltaW)

plt.plot(square_error)
plt.xlabel('No. of epoch')
plt.ylabel('Weight Change')
plt.title('Adaptive Linear Neuron - Learning Rate ' + str(alpha))
plt.show()

print(sum_squared_error)

# Prediction OR Gate
for j in range(features.shape[0]):
    x1 = features[j][0]
    x2 = features[j][1]
    
    unit = (x1 * weight[0]) + (x2 * weight[1]) + bias
    print(x1, ' ' , x2, ' ', unit>0)
