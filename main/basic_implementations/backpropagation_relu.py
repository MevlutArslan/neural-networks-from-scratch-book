import numpy as np
# Example layer output
z = np.array([[1, 2, -3, -4],
              [2, -7, -1, 3],
              [-1, 2, 5, -1]])

dvalues = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])

# ReLU activation's derivative
# fill an array with zeroes with the same shape as z
drelu = np.zeros_like(z)
# if index is greater than 0 than make it equal to corresponding derivative
drelu = dvalues.copy()
drelu[z <= 0] = 0

print(drelu)
