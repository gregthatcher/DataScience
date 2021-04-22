"""
This module displays graphs of the Bernoulli Distribution
The Bernoulli distribution only takes a paramter of mu (the average)
It is for getting a distribution for a single trial with two outcomes
"""
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# Possible stats.stackexhange.com question:
# What is intuition behind distributions getting narrower with smaller mu (average)
fig, ax = plt.subplots(1, 3, figsize=(10,7))

plt.style.use("ggplot")

fig.suptitle("Bernoulli Distribution: P(x; p) = (px) + (1-p)(1-x); "\
    "used for 1 trial with two outcomes.\n"\
    "x is 0 or 1; Var[X] = E[X^2]-E[X]^2")

x = np.arange(-1, 2, 0.5)
for index, p in enumerate([0.3, 0.5, 0.7]):
    ax[index].vlines(x, 0, stats.bernoulli.pmf(x, p), colors='b', lw=5, alpha=0.5)
    ax[index].set_title(f"p={p}")

plt.show()
