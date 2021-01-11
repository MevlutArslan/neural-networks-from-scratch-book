import numpy as np

layer_outputs = [4.8, 1.21, 2.385]

E = 2.71828182846
'''
    For each value in the vector calculate exponential value
    Vanilla Python
'''
exp_values = []
for output_value in layer_outputs:
    exp_values.append(E ** output_value)

# Normalize
norm_values = []
for value in exp_values:
    norm_values.append(value / sum(exp_values))

print(norm_values)
'''
    Numpy Version
    
    np.exp does E ** for every value in layer_outputs and returns a
    list containing the results in the same shape
'''
exp_values = np.exp(layer_outputs)


# Normalize
norm_values = exp_values / np.sum(exp_values)

print(norm_values)

''' 
    To train in batches, we need to convert this 
    functionality to accept layer outputs in batches.
'''

# Get unnormalized probabilities
exp_values = np.exp(inputs)
# Normalize them for each sample

'''
    axis : In a 2D array/matrix, axis 0 refers to the rows, and axis 1 refers to the columns.
    keepdims : Keeps the dimensions of the input.
'''

'''
 numpy will divide all of the values from each output row by the corresponding row from the sum array.
'''
probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
