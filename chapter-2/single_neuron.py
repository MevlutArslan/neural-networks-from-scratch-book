import random

# what goes into the neural network
inputs = [1.0, 2.0, 3.0, 2.5]

# FANCY RANDOMIZER
# weights = [random.uniform(0, 1) for x in range(len(inputs))]

# weights are random when initalizing the neuron
weights = [0.2, 0.8, -0.5, 1.0]
# Biases are set to 0 when initializing the neuron
# Since we’re modeling a single neuron, we only have one bias,
# as there’s just one bias value per neuron.
bias = 2

'''
    OUTPUT = SUM(INPUTS * WEIGHTS) + BIAS
'''
# single neuron


def get_output(inputs, weights, bias):
    output = 0
    for i in range(len(inputs)):
        output += inputs[i] * weights[i]

    output += bias
    return output


print(get_output(inputs, weights, bias))
