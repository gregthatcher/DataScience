import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# Possible stats.stackexhange.com question:
# What is intuition behind distributions getting narrower with smaller mu (average)
fig, ax = plt.subplots(2, 3, figsize=(10,10))

plt.style.use("ggplot")

#fig.suptitle(r"Poisson Distribution: P(x; μ) = ($\euler_-μ$) (μx) / x!")
fig.suptitle('Poisson Distribution: P(x; μ) = (e^μ)(μ^x) / x!; Variance = μ')

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


y = stats.poisson.pmf(x, mu=80)
ax[1][0].plot(x,y)
ax[1][0].set_title("mu=80")

y = stats.poisson.pmf(x, mu=40)
ax[1][1].plot(x,y)
ax[1][1].set_title("mu=40")

y = stats.poisson.pmf(x, mu=20)
ax[1][2].plot(x,y)
ax[1][2].set_title("mu=20")

plt.show()
