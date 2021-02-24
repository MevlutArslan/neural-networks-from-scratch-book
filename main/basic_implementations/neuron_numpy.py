from neuron import Neuron

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]

bias = 2


neuron = Neuron(inputs, weights, bias)

print(neuron.calculate_output_numpy())