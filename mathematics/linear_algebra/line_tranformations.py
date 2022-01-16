import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)

print(plt.style.available)
plt.style.use("seaborn-whitegrid")

x = np.linspace(0, 10, 101)
y = 1.5 * x

ax[0][0].plot(x, y)
ax[0][0].set_title("Original")

ax[0][1].plot(x, 2 * y)
ax[0][1].set_title("Multiply by 2")

ax[0][2].plot(x, y / 2)
ax[0][2].set_title("Divide by 2")

ax[1][0].plot(x, y + 3)
ax[1][0].set_title("Offset by 3")

ax[1][1].plot(x, y - 3)
ax[1][1].set_title("Offset by -3")

ax[1][2].plot(x, (2 * y) + 3)
ax[1][2].set_title("Multiply by 2 and add 3")

plt.tight_layout()
plt.show()
