from Classes.layer_dense import Layer_Dense
from Classes.relu_activation_func import ReLUActivation
from Classes.softmax_activation_func import SoftmaxActivation
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()

X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
# Perform a forward pass of our training data through this layer
dense1.forward(X)

activation1 = ReLUActivation()
activation2 = SoftmaxActivation()

activation1.forward(dense1.output)
activation2.forward(dense1.output)

print(dense1.output[:5])

print("Softmax ouput : \n")
print(activation2.output[:5])

print("RELU output : \n")
print(activation1.output[:5])
