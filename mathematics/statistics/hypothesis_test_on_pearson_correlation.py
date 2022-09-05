
import numpy as np
import matplotlib.pyplot as plt
import random


def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0, 1]


def compute_p_value(x, y):
    observed_correlation = pearson_r(x, y)
    print(f"Observed Correlation {observed_correlation}")

    number_of_replicates = 10000
    perm_replicates = np.empty(number_of_replicates)
    for i in range(number_of_replicates):
        y_permuted = np.random.permutation(y)
        perm_replicates[i] = pearson_r(x, y_permuted)

    p = np.sum(perm_replicates >= observed_correlation) / number_of_replicates
    if (p <= 0.05):
        msg = \
        f"P value {p};  Null Hypothesis fails,  variables are _not_ " \
        f"independent."
    else:
        msg = \
        f"P value {p};  Null Hypothesis succeeds,  variables are independent."

    return (p, msg)

x = np.arange(0, 100, 0.1)
y = 2 * x + (100 * np.random.rand(len(x)))

p, msg = compute_p_value(x, y)

plt.plot(x, y)
plt.title(msg)
plt.show()

x = np.arange(0, 100, 0.1)
y = np.random.rand(len(x))

p, msg = compute_p_value(x, y)

plt.plot(x, y)
plt.title(msg)
plt.show()
