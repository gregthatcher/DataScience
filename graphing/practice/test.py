import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2, 101)

fig, ax = plt.subplots()

plt.plot(x, x, label="linear")
plt.plot(x, x**2, label="quadratic")
plt.plot(x, x**3, label="cubic")
plt.legend()
plt.show()

functions = [func for func in dir(ax) if func.startswith("get")]
print(functions)