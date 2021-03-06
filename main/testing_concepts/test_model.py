import nnfs
from nnfs.datasets import spiral_data

import numpy as np
from main.classes.layer_dense import Layer_Dense
from main.classes.relu_activation_func import ReLUActivation
from main.classes.softmax_ccel_combined import Activation_Softmax_Loss_CategoricalCrossentropy
from main.classes.sgd_optimizer import SGD_Optimizer
from main.classes.adagrad_optimizer import Optimizer_Adagrad
from main.classes.rms_prop_optimizer import Rms_Prop_Optimizer
from main.classes.adam_optimizer import Adam_Optimizer
from main.classes.dropout_layer import Dropout_Layer

# Create dataset
X, y = spiral_data(samples=500, classes=3)

# X_test, y_test = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 16, weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4)

# Create ReLU activation (to be used with Dense layer):
activation1 = ReLUActivation()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(16, 16)

activation2 = ReLUActivation()

dense3 = Layer_Dense(16, 3)
# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# optimizer = SGD_Optimizer(learning_rate=1.0, decay=1e-3, momentum=0.9)
# optimizer = Optimizer_Adagrad(decay=1e-5)
# optimizer = Rms_Prop_Optimizer(learning_rate=0.02, decay=1e-4, rho=0.999)
optimizer = Adam_Optimizer(learning_rate=0.05, decay=5e-5)

dropout_layer = Dropout_Layer(0.1)

for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)
    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)

    dropout_layer.forward(activation1.output)
    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(dropout_layer.output)

    activation2.forward(dense2.output)

    dense3.forward(activation2.output)

    # Calculate loss from output of activation2 so softmax activation
    # data_loss = loss_activation.forward(dense2.output, y)
    # # Calculate regularization penalty
    # regularization_loss = \
    #     loss_activation.loss.regularization_loss(dense1) \
    #     + loss_activation.loss.regularization_loss(dense2)

    # Calculate overall loss
    # loss = data_loss + regularization_loss

    loss = loss_activation.forward(dense3.output, y)

    # Calculate accuracy from output of activation2 and targets # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
        f'acc: {accuracy:.3f}, ' +
        f'loss: {loss:.3f} (' + f'lr: {optimizer.current_learning_rate}')
    
    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense3.backward(loss_activation.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)

    dropout_layer.backward(dense2.dinputs)
    activation1.backward(dropout_layer.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_parameters(dense1)
    optimizer.update_parameters(dense2)
    optimizer.post_update_params()

# Create test dataset
X_test, y_test = spiral_data(samples=500, classes=3)
# Perform a forward pass of our testing data through this layer 
dense1.forward(X_test)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2.forward(dense2.output)

dense3.forward(activation2.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
# dense2.forward(activation1.output)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense3.output, y_test)
# Calculate accuracy from output of activation2 and targets # calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1) 
accuracy = np.mean(predictions == y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

