from Classes.neuron import Neuron

inputs = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

neurons = [
    Neuron(inputs, [0.2, 0.8, -0.5, 1], 2),
    Neuron(inputs, [0.5, -0.91, 0.26, -0.5], 3),
    Neuron(inputs, [-0.26, -0.27, 0.17, 0.87], 0.5)
]

for neuron in neurons:
    print(neuron.calculate_output())
