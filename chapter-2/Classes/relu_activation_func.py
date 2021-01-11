import numpy as np


class ReLUActivation():
    # Forward pas
    def forward(self, inputs):
        # Calculate output values from input
        self.output = np.maximum(0, inputs)
