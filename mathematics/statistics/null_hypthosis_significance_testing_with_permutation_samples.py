# Ideas from : https://campus.datacamp.com/courses/statistical-thinking-in-python-part-2/bootstrap-confidence-intervals?ex=14
# Our "null hypothesis" is that two samples are the same
# We "permute" the two samples, keeping the two original sample sizes,
# and then test to see if our "test statistic" is invariant
# under the new "permuted" data.
# This is sometimes referred to as "Null Hypothesis Significance Testing"

import numpy as np
import matplotlib.pyplot as plt
from ecdf import ecdf


def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate([data1, data2])

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2


# Not currently using this function, but got it
# from https://campus.datacamp.com/courses/statistical-thinking-in-python-part-2/introduction-to-hypothesis-testing?ex=7
def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates


def show_data(data1, data2, title=""):
    for _ in range(50):
        # Generate permutation samples
        perm_sample_1, perm_sample_2 = permutation_sample(
            first_fake_data, second_fake_data)

        # Compute ECDFs
        x_1, y_1 = ecdf(perm_sample_1)
        x_2, y_2 = ecdf(perm_sample_2)

        # Plot ECDFs of permutation sample
        _ = plt.plot(x_1, y_1, marker='.', linestyle='none',
                     color='red', alpha=0.02)
        _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                     color='blue', alpha=0.02)

    # Create and plot ECDFs from original data
    x_1, y_1 = ecdf(data1)
    x_2, y_2 = ecdf(data2)
    _ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
    _ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

    # Label axes, set margin, and show plot
    plt.margins(0.02)
    _ = plt.xlabel('monthly rainfall (mm)')
    _ = plt.ylabel('ECDF')
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    data_size = 100
    first_fake_data = (2 + np.random.random(size=data_size)) * \
        + 4 + \
        21 * np.random.random(size=data_size)

    # Data are much different
    second_fake_data = first_fake_data + 10 * \
        np.random.random(size=len(first_fake_data))

    show_data(first_fake_data, second_fake_data, "Data are Different")

    # data are not so different
    second_fake_data = first_fake_data + 1.0 * \
        np.random.random(size=len(first_fake_data)) - 0.5
    show_data(first_fake_data, second_fake_data, "Data are Similar")
