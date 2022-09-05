import math
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    #ret = math.exp(x) / math.exp(x+1)
    return ret


x = np.arange(-5, 5, 0.1)
#y = np.exp(x) / (np.exp(x) + 1)
y = np.exp(x)

plt.plot(x, y)
plt.show()
sigmoid(-0.1)
sigmoid(-0.5)
sigmoid(-1)
print(x)

print(math.exp(-1))

