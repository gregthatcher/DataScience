# Ideas from : https://campus.datacamp.com/courses/statistical-thinking-in-python-part-1/graphical-exploratory-data-analysis?ex=12

import numpy as np
import matplotlib.pyplot as plt


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y


if __name__ == "__main__":
    y = np.random.normal(0, 1, size=2000)

    x, y = ecdf(y)
    plt.plot(x, y)
    plt.ylabel("ECDF")

    plt.show()
