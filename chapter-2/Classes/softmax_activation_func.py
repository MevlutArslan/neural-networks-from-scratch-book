
class SoftmaxActivation():
    # Forward pass

    def forward(self, inputs):
        # Get unnormalized probabilities
        '''
            We also included a subtraction of the largest of the inputs before we did the exponentiation.
            With Softmax, thanks to the normalization, we can subtract any value from all of the inputs, 
            and it will not change the output.
        '''
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
