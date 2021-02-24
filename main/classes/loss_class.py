import numpy as np

# Common loss class
class Loss:
    # Calculates the data and regularization losses # given model output and ground truth values def calculate(self, output, y):
    # Calculate sample losses
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss
    
    def regularization_loss(self, layer):
        regularization_loss = 0

        if layer.weight_regularizer_l1 > 0 :
            regularization_loss += layer.weight_regularizer_l1 + \
                                    np.sum(np.abs(layer.weights))

        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 + \
                                    np.sum(layer.weights * layer.weights)

        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 + \
                                    np.sum(np.abs(layer.biases))

        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 + \
                                    np.sum(layer.biases * layer.biases)

        return regularization_loss
