import numpy as np
import math as mt

def sigmoid( mat):
    temp = mat.copy()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            temp[i][j] = 1.0 / ( 1 + mt.exp(-mat[i][j]))
    
    return temp

# Features
# XOR Model
x = np.array([
    [-1, -1],
    [-1, 1],
    [1, -1],
    [1, 1]
])

# Target
y = np.array([-1, 1, 1, -1])

# Initialization
# Weights are randomly assigned between [-0.5, 0.5]
a = -0.5
b = 0.5
wij = ( b - a ) * np.random.random_sample((3,2)) + a
print(f'\n Randomly initialized weight for input layer \n {wij}')

wjk = (b - a ) * np.random.random_sample((3,1)) + a
print(f'\n Randomly initialized weight for outputlayer \n {wjk}')


# Bias 

bias_in = np.array([[1],[1]])
bias_out = np.array([1])

# Learning rate
alpha = 0.01
epoch = 100

for epo in range(epoch):
    print(f'Epoch : { epo + 1 } \n\n')

    # for each of the possible input given in the features
    for j in range(x.shape[0]):
        xin = np.array([[bias_in[0][0], x[j][0], x[j][1]]])
        xin = xin.transpose()

        z = (wij.transpose()) @ xin
        tp = sigmoid(z)

        zin = np.array([bias_out[0], tp[0][0], tp[1][0]])
        zin = zin.reshape(3,1)

        yin_k = (wjk.transpose()) @ zin
        yin_k = yin_k.reshape(1,1)

        yk = sigmoid(yin_k)

        del_k = ( y[0] - yk) * yk * ( 1 - yk)

        del_wjk = alpha * del_k * zin

        del_inj = del_k * wjk

        del_j = del_inj * zin * (1 - zin)
        



