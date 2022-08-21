# Ideas from https://www.youtube.com/watch?v=UetYS3PaHIo

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

x = np.arange(-5, 5, 0.1)

mu = 0
std = 1
normal_y = stats.norm.pdf(x, mu, std)
# One degree of freedome (2 samples)
t_1 = stats.t(df=1, loc=mu, scale=std)
# 5 degrees of freedom (6 samples)
t_5 = stats.t(df=5, loc=mu, scale=std)
# 20 degrees of freedom
t_20 = stats.t(df=20, loc=mu, scale=std)

plt.plot(x, normal_y, color="red", label="normal")
plt.plot(x, t_1.pdf(x), label="1 degree of freedom",
         marker=".", linestyle="none")
plt.plot(x, t_5.pdf(x), label="5 degrees of freedom",
         marker=".", linestyle="none")
plt.plot(x, t_20.pdf(x), label="20 degrees of freedom",
         marker=".", linestyle="none")
plt.text(1, 0.35,
         "As degrees of freedom increases, t-dist becomes normal dist.",
         wrap=True)
plt.text(1, 0.30,
         "t-dist compensates for low sample size by increasing std dev.",
         wrap=True)
plt.legend()

plt.show()
