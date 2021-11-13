'''
Ideas from : https://www.youtube.com/watch?v=8krd5qKVw-Q&list=PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU&index=31
This is a simulation to show that the sum of n variables with variance
1 has a variance of n * original variance.
This is important to know as it explains why we need
"weight initialization" to avoid instability in gradient descent.
Note that instability in gradient descent can lead to the
vanishing gradient problem (weights in first layers get very small updates)
or the exploding gradient problem (weights in first layers are updated too
much).
Since a node is getting multiple inputs (each with its own variance),
the variance of their sum has a variance which is the sum of the
different variances.
'''

import numpy as np

# Note that np.random() generates numbers in range [0, 1] from
# uniform distribution

k_num_elements = 10000

one_variable = np.random.rand(k_num_elements)
print(one_variable)
print(f"Variance of one variable: {np.var(one_variable):.3}")


def calc_variance(num_variables):
    vector = np.random.rand(k_num_elements)
    original_var = np.var(vector)
    for i in range(num_variables-1):
        vector += np.random.rand(k_num_elements)

    print(f"Variance of sum of {num_variables} variables {np.var(vector):.3}")
    print(f"Increase by factor of {np.var(vector)/original_var:.3f}")  


calc_variance(1)
calc_variance(10)
calc_variance(100)
calc_variance(1000)
