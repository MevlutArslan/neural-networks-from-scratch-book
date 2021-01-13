import numpy as np


class ReLUActivation():

    def forward(self, inputs):
        # Calculate output values from input
        '''
            The ReLU activation function outputs 0 for all inputs thats less than 0 and
            it outputs the input for all values greater or equal to zero.
        '''
        self.output = np.maximum(0, inputs)
