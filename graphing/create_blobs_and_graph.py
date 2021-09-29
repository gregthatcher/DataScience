'''
How to create and graph a couple of blobs
See also: separate_blobs_with_neural_network.py
'''

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs


X, y = make_blobs(n_samples=1000, centers=2, random_state=42)

plt.plot(X[y == 0, 0], X[y == 0, 1], "ob", alpha=0.5)
plt.plot(X[y == 1, 0], X[y == 1, 1], "xr", alpha=0.5)
plt.legend(["0", "1"])

plt.show()
