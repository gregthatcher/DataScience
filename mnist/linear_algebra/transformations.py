import math
import numpy as np
import matplotlib.pyplot as plt

def show_subplot(ax, up, right, transform, 
        title, text, text_x, text_y):    
    vector_1 = transform @ up
    vector_2 = transform @ right
    ax.plot(vector_1[0], vector_1[1])
    ax.plot(vector_2[0], vector_2[1])
    ax.set_title(title)
    ax.text(text_x, text_y, text)

num_rows = 3
num_columns = 3
fig, ax = plt.subplots(num_rows, num_columns, sharex=True, sharey=True)

plt.style.use("ggplot")

up = np.array([0, 0, 0, 1])
up = up.reshape(2, 2)
right = np.array([0, 1, 0, 0])
right = right.reshape(2, 2)

for i in range(num_rows):
    for j in range(num_columns):
        ax[i][j].set_xlim(-1.1, 1.1)
        ax[i][j].set_ylim(-1.1, 1.1)

ax[0][0].plot(up[0], up[1])
ax[0][0].plot(right[0], right[1])

show_subplot(ax[0][0], up, right, 
    np.array([1,0,0,1]).reshape(2, 2), 
    "Original", "", -0.9, 0.5)

show_subplot(ax[0][1], up, right, 
    np.array([0,1,1,0]).reshape(2, 2), 
    "Reflect over y=x", "0,1\n1,0", -0.9, 0.5)

show_subplot(ax[0][2], up, right, 
    np.array([0.5, 0, 0, 0.5]).reshape(2, 2), 
    "Scaling by 1/2", "0.5,0\n0,0.5", -0.9, 0.5)

show_subplot(ax[1][0], up, right, 
    np.array([1, 0.5, 0, 1]).reshape(2, 2), 
    "X Shear", "1,0.5\n0,1", -0.9, 0.5)

show_subplot(ax[1][1], up, right, 
    np.array([1, 0, .5, 1]).reshape(2, 2), 
    "Y Shear", "1,0\n.5,1", -0.9, 0.5)

angle = math.radians(30)

show_subplot(ax[1][2], up, right, 
    np.array([math.cos(angle), math.sin(angle), -math.sin(angle), math.cos(angle)]).reshape(2, 2), 
    "Rotate by 30 degrees", "", -0.9, 0.5)

plt.show()
