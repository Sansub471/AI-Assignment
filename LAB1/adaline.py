import numpy as np
from matplotlib import pyplot as plt

# OR Gate 
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
weight = [-0.2, 0.3] # w1 w2
bias = 1 # b
alpha = 0.01 # alpha
epoch = 1000

weight_tolerance = 0.008 # If weight change is less than this, algorithm terminates.
weight_change = []

def display(num):
    print(f'----------Epoch : {num} ------------------------')
    print(f'Weight [w1, w1] : {weight}')
    print(f'Bias : {bias}')

print("Initialized values")
display(0)

for i in range(epoch):
    
    training_set_weight_change = []
    # for each of the possible input given in the features
    for j in range(features.shape[0]):
  
        # actual output to be obtained
        actual = target[j]
  
        # the value of two features as given in the features array
        x1 = features[j][0]
        x2 = features[j][1]
  
        # Yin
        yin = (x1 * weight[0]) + (x2 * weight[1]) + bias
        
        # ( t - Yin)
        error = actual - yin
        
        delW1 = x1 * alpha * error
        delW2 = x2 * alpha * error
        deltaW = max(delW1, delW2)
        training_set_weight_change.append(deltaW)
        
  
        # update weights
        weight[0] += delW1
        weight[1] += delW2
        

        # update bias       
        bias += alpha * error
        
    epoch_delW = max(training_set_weight_change)
    weight_change.append(epoch_delW)
    display(i+1)
    
    if epoch_delW < weight_tolerance:
        break

print(f'Minimum weight change : {min(weight_change)}')

plt.plot(weight_change)
plt.xlabel('No. of epoch')
plt.ylabel('Weight Change')
plt.title('Adaptive Linear Neuron - Learning Rate ' + str(alpha))
plt.show()

# Prediction OR Gate
for j in range(features.shape[0]):
    x1 = features[j][0]
    x2 = features[j][1]
    
    unit = (x1 * weight[0]) + (x2 * weight[1]) + bias
    print(x1, ' ' , x2, ' ', unit>0)
