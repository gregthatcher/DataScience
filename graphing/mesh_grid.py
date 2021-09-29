'''
How to draw a "mesh" of points
Ideas from https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-python-numpy
'''

import numpy as np
import matplotlib.pyplot as plt

xvalues = np.array([0, 1, 2, 3, 4])
yvalues = np.array([0, 1, 2, 3, 4])

xx, yy = np.meshgrid(xvalues, yvalues)

print("X:")
print(xx)

print("Y:")
print(yy)
plt.plot(xx, yy, marker='.', color='k', linestyle='none')
plt.show()
