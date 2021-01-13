import nnfs
import numpy as np
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
        self.output = np.dot(inputs, self.weights) + self.biases


class ReLUActivation():

    def forward(self, inputs):
        # Calculate output values from input
        '''
            The ReLU activation function outputs 0 for all inputs thats less than 0 and
            it outputs the input for all values greater or equal to zero.
        '''
        self.output = np.maximum(0, inputs)


class SoftmaxActivation():
    # Forward pass

    def forward(self, inputs):
        # Get unnormalized probabilities
        '''
            We also included a subtraction of the largest of the inputs before we did the exponentiation.
            With Softmax, thanks to the normalization, we can subtract any value from all of the inputs, 
            and it will not change the output.

            Large values can cause serious problems in Neural Networks and because the output does not change
            whenever we subtract any value from the inputs it is a good idea to get the numbers as small as possible.

            The reason subtracting a value from inputs does not change anything is because all of the outputs 
            after running Softmax Activation Function will add up to one. It works with proportions and because
            we are subtracting the same amount for all inputs the dividing any number with any other number will
            be the same amount.
        '''
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = ReLUActivation()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

# Create Softmax activation (to be used with Dense layer):
activation2 = SoftmaxActivation()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)

# This is what goes into the softmax function, it creates the exponential of it.
# 
print((activation1.output - np.max(activation1.output))[:5])
print(activation2.output[:5])
