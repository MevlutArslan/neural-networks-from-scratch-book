import nnfs
from nnfs.datasets import spiral_data

import numpy as np
from layer_dense import Layer_Dense
from relu_activation_func import ReLUActivation
from softmax_ccel_combined import Activation_Softmax_Loss_CategoricalCrossentropy
from sgd_optimizer import SGD_Optimizer
from adagrad_optimizer import Optimizer_Adagrad
from rms_prop_optimizer import Rms_Prop_Optimizer
from adam_optimizer import Adam_Optimizer

# Create dataset
X, y = spiral_data(samples=100, classes=3)

X_test, y_test = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 64)

# Create ReLU activation (to be used with Dense layer):
activation1 = ReLUActivation()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(64, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# optimizer = SGD_Optimizer(learning_rate=1.0, decay=1e-3, momentum=0.9)
# optimizer = Optimizer_Adagrad(decay=1e-5)
# optimizer = Rms_Prop_Optimizer(learning_rate=0.02, decay=1e-4, rho=0.999)
optimizer = Adam_Optimizer(learning_rate=0.05, decay=1e-6)

for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X_test)
    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)
    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)

    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = loss_activation.forward(dense2.output, y_test)

    # Calculate accuracy from output of activation2 and targets # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f},' +
              f'lr: {optimizer.current_learning_rate}')

    # Backward pass
    loss_activation.backward(loss_activation.output, y_test)
    dense2.backward(loss_activation.dinputs)

    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_parameters(dense1)
    optimizer.update_parameters(dense2)
    optimizer.post_update_params()

# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)
  # Perform a forward pass through second Dense layer
  # takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
  # Perform a forward pass through the activation/loss function
  # takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y_test)
# Calculate accuracy from output of activation2 and targets # calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1) 
accuracy = np.mean(predictions == y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')