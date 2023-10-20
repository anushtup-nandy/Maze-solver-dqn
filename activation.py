import numpy as np
from config import learningCoreSettings
import torch
import matplotlib.pyplot as plt

#WEIGHTS:
def initWeight(intSize, outSize):
    W=np.random.randn(intSize, outSize)/np.sqrt(intSize)  #randomly initialising weights from a normal distribution
    b=np.zeros(outSize)
    return W.astype(np.float32), b.astype(np.float32)

#ACTIVATION FUNCTIONS FOR THE NEURAL NETWORK:
# Relu activation function
def relu(x):
    return x * (x > 0)

# Sigmoid activation function
def sigmoid(A):
    return 1 / (1 + np.exp(-A))

# Softmax layer implementation
def softmax(A):
    expA = np.exp(A/learningCoreSettings["softmaxTemperature"])
    return expA / expA.sum()




