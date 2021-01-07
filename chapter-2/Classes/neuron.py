
class Neuron():
    def __init__(self, inputs: list, weights: list, bias: float):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias

        self.number_of_inputs = len(inputs)

    def calculate_output(self):
        output = 0
        for i in range(self.number_of_inputs):
            inpt = self.inputs[i]
            weight = self.weights[i]

            output += inpt * weight
        
        output += self.bias

        return output
