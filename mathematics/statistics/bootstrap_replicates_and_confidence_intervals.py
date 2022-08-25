# Ideas from : https://campus.datacamp.com/courses/statistical-thinking-in-python-part-2/bootstrap-confidence-intervals?ex=6

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def bootstrap_replicate_1d(data, func):
    """ Generate boostrap replicate of 1D data """
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)


def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates


fake_data = np.arange(0, 101, 1)

fig, ax = plt.subplots()

# Compute "Standard Error of Mean"
# Remember that mean is Normally distributed (normally)
# and this is a calculation of the standard deviation
# of that distribution.
sem = np.std(fake_data) / np.sqrt(len(fake_data))
ax.text(40, 0.16, f"Standard Error of Mean {sem:.2f} (std dev of this graph)")

bs_replicates = draw_bs_reps(fake_data, np.mean, size=10000)

# Calculate "confidence interval"
# We expect 95% of data values in this range
conf_int = np.percentile(bs_replicates, [2.5, 97.5])
ax.text(40, 0.15,
        f'95% confidence interval between {conf_int[0]:.2f}'
        f' and {conf_int[1]:.2f}')

ax.hist(bs_replicates, bins=30, density=True)
ax.set_xlabel("mean of fake data")
ax.set_ylabel("PDF")

# ax.vlines(conf_int[0], 0, 0.14, colors="red")
# ax.vlines(conf_int[1], 0, 0.14, colors="red")

rect = patches.Rectangle((conf_int[0], 0), conf_int[1] - conf_int[0], 0.141,
                         linewidth=1, edgecolor="red", facecolor="none")
ax.add_patch(rect)

plt.show()
