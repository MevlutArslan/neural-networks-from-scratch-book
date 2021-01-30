import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Forward pass

    def forward(self, inputs):
        # we store cause : 
        # we’ll need them when calculating the partial derivative with respect to weights during backpropagation
        self.inputs = inputs 
        self.output = np.dot(inputs, self.weights) + self.biases

    # method to calculate the gradient,
    # takes in the derivates from the previous step in back propagation
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.weights.T)

        