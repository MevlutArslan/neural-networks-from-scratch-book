'''
The first option one might think of is randomly 
changing the weights, checking the loss, and 
repeating this until happy with the lowest loss found.
'''

import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

import numpy as np

from layer_dense import Layer_Dense
from relu_activation_func import ReLUActivation
from softmax_activation_func import SoftmaxActivation

from cross_entropy import Loss_CategoricalCrossentropy
nnfs.init()

coords, classes = spiral_data(samples=100, classes=3)


dense_1 = Layer_Dense(2, 3)

activation_1 = ReLUActivation()

dense_2 = Layer_Dense(3, 3)

activation2 = SoftmaxActivation()

loss_function = Loss_CategoricalCrossentropy()

lowest_loss = 9999999

best_dense1_weights = dense_1.weights.copy()
best_dense1_biases = dense_1.biases.copy()
best_dense2_weights = dense_2.weights.copy()
best_dense2_biases = dense_2.biases.copy()

for iteration in range(100000):

    # Generate a new set of weights for iteration
    dense_1.weights += 0.05 * np.random.randn(2, 3)
    dense_1.biases += 0.05 * np.random.randn(1, 3)
    dense_2.weights += 0.05 * np.random.randn(3, 3)
    dense_2.biases += 0.05 * np.random.randn(1, 3)

    # Perform a forward pass of the training data through this layer
    dense_1.forward(coords)
    activation_1.forward(dense_1.output)

    dense_2.forward(activation_1.output)
    activation2.forward(dense_2.output)

    # Perform a forward pass through activation function
    # it takes the output of second dense layer here and returns loss
    loss = loss_function.calculate(activation2.output, classes)

    # Calculate accuracy from output of activation2 and targets # calculate values along first axis
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == classes)

    # If loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        print('New set of weights found, iteration:', iteration,
              'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense_1.weights.copy()
        best_dense1_biases = dense_1.biases.copy()
        best_dense2_weights = dense_2.weights.copy()
        best_dense2_biases = dense_2.biases.copy()
        lowest_loss = loss
    else:
        dense_1.weight = best_dense1_weights.copy()
        dense_1.biases = best_dense1_biases.copy()
        dense_2.weights = best_dense2_weights.copy()
        dense_2.biases = best_dense2_biases.copy()
