'''
The example below shows the difference in t-distributions as we change the
"degrees of freedom"
Ideas from : https://pub.towardsai.net/fully-explained-t-distribution-with-python-example-b861413ceb9
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.arange(-4, 4, 0.1)
t_one_degree = stats.t.pdf(x, 1)
t_three_degrees = stats.t.pdf(x, 3)
t_thirty_degrees = stats.t.pdf(x, 30)
normal = stats.norm.pdf(x)

plt.plot(x, t_one_degree, label="1 degree of freedom")
plt.plot(x, t_three_degrees, label="3 degrees of freedom")
plt.plot(x, t_thirty_degrees, label="30 degrees of freedom")
plt.legend()
plt.title("t-distributions with different degrees of freedom")
plt.show()

plt.plot(x, t_one_degree, label="1 degree of freedom")
plt.plot(x, t_three_degrees, label="3 degrees of freedom")
plt.plot(x, t_thirty_degrees, label="30 degrees of freedom")
plt.plot(x, normal, label="Normal Distribution")
plt.legend()
plt.title("Compare t-distributions with normal distribution")
plt.show()
