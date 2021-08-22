'''
Showing some plots of normal distributions using various libraries
'''

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

X = np.arange(-10, 10, 0.1)

y = stats.norm.pdf(X)

plt.plot(X, y)
plt.title("Normal Distribution using scipy")
plt.show()

sns.distplot(np.random.normal(size=10000), hist=False)
plt.title("Distribution plot using Seaborn")
plt.show()
