'''
Draw two concentric circles
Ideas from https://app.pluralsight.com/course-player?clipId=17dea10c-ddbf-4866-b6b2-b1f88224d6cd
'''

import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

# Factor -> Scale factor between inner and outer circle in the range (0, 1)
# Noise  -> Standard deviation of Gaussian noise added to the data. 
X, y = make_circles(n_samples=1000, factor=.6, noise=0.1, random_state=42)

plt.plot(X[y == 0, 0], X[y == 0, 1], "ob", alpha=0.5)
plt.plot(X[y == 1, 0], X[y == 1, 1], "xr", alpha=0.5)
plt.legend(["0", "1"])

plt.show()
