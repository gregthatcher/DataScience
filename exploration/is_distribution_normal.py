'''
Test to see if a distribution is normal or not
'''
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.gofplots import qqplot

fig, ax = plt.subplots(2, 3, figsize=(12,10))

mu, sigma = 0, 0.1 # mean and standard deviation
data_size = 1000
normal_data = np.random.normal(mu, sigma, data_size)
uniform_data = np.random.uniform(size=data_size)

ax[0][0].plot(np.arange(len(normal_data)), normal_data)
ax[0][0].set_title("Plot of Random Normal Data")

ax[0][1].hist(normal_data)
ax[0][1].set_title("Histogram of Normal Data")

# This site also has other stats we might try
# https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
shapiro_calcs = shapiro(normal_data)
ax[0][2].text(-2, 0.19, f"Statistic: {round(shapiro_calcs.statistic, 2)}")
ax[0][2].text(-2, 0.1, f"p-value: {round(shapiro_calcs.pvalue, 2)}")
qqplot(normal_data, line="s", ax=ax[0][2])
ax[0][2].set_title("QQ Plot/Shapiro Stats for Normal Data")

ax[1][0].plot(np.arange(len(uniform_data)), uniform_data)
ax[1][0].set_title("Plot of Random Uniform Data")

ax[1][1].hist(uniform_data)
ax[1][1].set_title("Histogram of Uniform Data")

shapiro_calcs = shapiro(uniform_data)
ax[1][2].text(-2, 1.2, f"Statistic: {round(shapiro_calcs.statistic, 2)}")
ax[1][2].text(-2, 1.0, f"p-value: {round(shapiro_calcs.pvalue, 2)}")
qqplot(uniform_data, line="s", ax=ax[1][2])
ax[1][2].set_title("QQ Plot/Shapiro Stats for Uniform Data")

plt.show()
