import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

num_rows = 2
num_columns = 3
fig, ax = plt.subplots(num_rows, num_columns, figsize=(10,10))

plt.style.use("ggplot")

x = np.arange(0, 100, 0.5)
y = stats.poisson.pmf(x, mu=40)

ax[0][0].plot(x,y)
ax[0][0].set_title("mu=40")

y = stats.poisson.pmf(x, mu=40, loc=10)
ax[0][1].plot(x,y)
ax[0][1].set_title("mu=40; loc=10")

y = stats.poisson.pmf(x, mu=10, loc=10)
ax[0][2].plot(x,y)
ax[0][2].set_title("mu=10; loc=10")

plt.show()
