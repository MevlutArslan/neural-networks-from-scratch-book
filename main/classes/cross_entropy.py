import numpy as np
from main.classes.loss_class import Loss

class Loss_CategoricalCrossentropy(Loss):  # Forward pass

    def forward(self, y_pred, y_true):  # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), y_true]
          # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped*y_true, axis=1)
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, ground_truth_v):

        sample_count = len(dvalues)

        label_count = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(ground_truth_v.shape) == 1:
            ground_truth_v = np.eye(label_count)[ground_truth_v]

        self.dinputs = -ground_truth_v / dvalues

        self.dinputs = self.dinputs / sample_count

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