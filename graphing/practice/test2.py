import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(0, 1, 101)
print(X)
plt.plot(X, 1/X, label="1/X")
plt.plot(X, -np.log2(X), label="-log(X)")
# plt.plot(X, np.log2(X), label="log(X)", linestyle="--")
plt.legend()
plt.show()
