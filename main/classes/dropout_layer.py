
import numpy as np

class Dropout_Layer:
    def __init__(self, drop_rate) -> None:
        self.drop_rate = 1 - drop_rate

    def forward(self, inputs):
        self.inputs = inputs

        self.binary_mask = np.random.binomial(1, self.drop_rate, inputs.shape) \
                         / self.drop_rate
        
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask