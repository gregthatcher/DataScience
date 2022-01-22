'''
Graphing Stock Data
Ideas from : https://towardsdatascience.com/6-python-matplotlib-features-to-create-better-data-visualizations-b9fc65b0430b
'''

from typing import FrozenSet
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt


plt.style.use("seaborn-darkgrid")

aapl = data.DataReader(
    "AAPL",
    start='2021-10-01',
    end='2021-12-31',
    data_source="yahoo"
)

plt.figure(figsize=(12, 6))
plt.plot(aapl["Close"])

plt.ylabel("Closing Price", fontsize=15)
plt.title("Apple Stock Price", fontsize=18, color="r")
plt.show()

fig, ax1 = plt.subplots(figsize=(12, 6))
# second Axes object
ax2 = ax1.twinx()
ax1.plot(aapl["Close"])
ax2.plot(aapl["Volume"], color="r")
# axis labels and title
ax1.set_ylabel("Closing Price", fontsize=15)
ax2.set_ylabel("Volume", fontsize=15)
plt.title("Apple Stock Price", fontsize=18)

# add legends
# The loc parameter specifies the position of the legend and
# 2 means “upper-left”.
ax1.legend(["Closing price"], loc=2, fontsize=12)
ax2.legend(["Volume"], loc=2, bbox_to_anchor=(0, 0.9), fontsize=12)

# remove grid
ax1.grid(False)
ax2.grid(False)

# add text - Not Working
# ax1.ottext(
#    "2021-10-31",
#    170,
#    "Nice plot!",
#    fontsize=18,
#    color="green"
#)

# tick size
ax1.tick_params(axis='both', which='major', labelsize=13)
ax2.tick_params(axis='both', which='major', labelsize=13)

plt.show()
