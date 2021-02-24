import numpy as np


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

    def backward(self, dvalues):

        # will create a new array with the same shape(dimensions)
        self.dinputs = np.empty_like(dvalues)

        # New trick, using \ at the end of the for loop allows the for loop to
        # continue on the next line :)
        '''
            enumerate returns a counter and the corresponding value
            in our case it will return an iterator and a tuple containing 
            self.output and dvalues
        '''
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):

            # flattens the array
            single_output = single_output.reshape(-1, 1)

            # we flattened the array to be able to use it in the below
            # calculation

            # calculates the jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)
