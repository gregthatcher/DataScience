import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-10, 10, 101)

fig, ax = plt.subplots(3, 4, figsize=(10, 8))

ax[0][0].plot(x, np.square(x))
ax[0][0].set_title("y = x^2")

ax[0][1].plot(x, np.sqrt(x))
ax[0][1].set_title("y = sqrt(x)")

ax[0][2].plot(x, np.log(x))
ax[0][2].set_title("y = ln(x)")
ax[0][2].axhline(y=0, c="r", linestyle="--")

ax[0][3].plot(x, np.log10(x))
ax[0][3].set_title("y = log10(x)")
ax[0][3].axhline(y=0, c="r", linestyle="--")

ax[1][0].plot(x, np.exp(x))
ax[1][0].set_title("y = exp(x)")

ax[1][1].plot(x, np.power(10, x))
ax[1][1].set_title("y = 10^x")

ax[1][2].plot(x, -np.log(x))
ax[1][2].set_title("y = -ln(x)")
ax[1][2].axhline(y=0, c="r", linestyle="--")

ax[1][3].plot(x, -np.log10(x))
ax[1][3].set_title("y = -log10(x)")
ax[1][3].axhline(y=0, c="r", linestyle="--")

ax[2][0].plot(x, 1/x)
ax[2][0].set_title("y = 1/x")
ax[2][0].axhline(y=0, c="r", linestyle="--")

plt.tight_layout()
plt.show()
