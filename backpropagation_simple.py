import numpy as np
# Passed-in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array(
    [
        [1., 1., 1.],
        [2., 2., 2.],
        [3., 3., 3.]
    ]
)

# We have 3 sets of inputs - samples
inputs = np.array(
    [
        [1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2],
        [-1.5, 2.7, 3.3, -0.8]
    ]
)


# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array(
    [
        [0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]
).T


# One bias for each neuron
# biases are the row vector with a shape (1, neurons)
biases = np.array([[2, 3, 0.5]])

# Sum weights related to the given input multiplied by
# the gradient related to the given neuron
'''
    The shape of the weights[0] is 3 rows and 1 column,
    the shape of dvalues[0] is 1 row 3 columns
    hence the rule applies for multiplying the matrices

    3x1 * 1x3 == Possible

   [0.2              [1.0    => .2
    0.5       *       1.0    => .5     => .44
   -0.26]             1.0]   => -.26

'''
dinputs = np.dot(dvalues, weights.T)

dweights = np.dot(inputs.T, dvalues)

# dbiases - sum values, do this over samples (first axis), 
# keepdims since this by default will produce a plain list
dbiases = np.sum(dvalues, axis=0, keepdims=True)

print(dinputs)

print(dweights)
