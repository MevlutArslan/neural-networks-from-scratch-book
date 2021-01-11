from Classes.layer_dense import Layer_Dense
from Classes.relu_activation_func import ReLUActivation
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()

X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
# Perform a forward pass of our training data through this layer
dense1.forward(X)

activation1 = ReLUActivation()

activation1.forward(dense1.output)

print(dense1.output[:5])
print(activation1.output[:5])
