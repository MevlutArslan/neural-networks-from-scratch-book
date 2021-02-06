import numpy as np
from softmax_activation_func import SoftmaxActivation
from cross_entropy import Loss_CategoricalCrossentropy


#Calculating the gradients separately is about 7 times slower
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = SoftmaxActivation()
        self.loss = Loss_CategoricalCrossentropy()

    # y_true is the ground truths
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)


    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2 :
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        '''
            Calculating the gradient based on the combines calculation of
            Categorical Cross-Entropy Loss function and Softmax Activation function
            which is the subtraction of the predicted and ground truth values.
        '''
        '''
            We’re taking advantage of the fact that the y being y_true 
            in the code consists of one-hot encoded vectors, which means that, 
            for each sample, there is only a singular value of 1 in these vectors 
            and the remaining positions are filled with zeros.
        '''
        self.dinputs[range(samples), y_true] -= 1
        # Normalize the gradient
        self.dinputs = self.dinputs / samples