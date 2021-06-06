'''
Uniform Continous Probablity
Given a range, (a,b), all values in (a,b] are equally likely
Height of PDF is 1/(b-a)
Ideas from: https://www.datacamp.com/community/tutorials/probability-distributions-python
'''
from scipy.stats import uniform
import matplotlib.pyplot as plt
import seaborn as sns

n = 10000
start = 10
width = 20
data_uniform = uniform.rvs(size=n, loc=start, scale=width)

plt.style.use("fivethirtyeight")

fig, ax = plt.subplots(2,1)

fig.suptitle("Uniform (Continuous) Distribution: f(x) = 1/(a+b)\n"
             "for a<=x<=b; zero otherwise\n"
             "All values between (a, b] are equally likely.\n"
             "E(x) = 1/2(a+b)\n"
             "Variance 1/12(b-a)\u00b2")

ax[0].plot(data_uniform)
ax[0].set_title("Raw Data")


sns.distplot(data_uniform,
             bins=100,
             kde=True,
             color='skyblue',
             hist_kws={"linewidth": 15, 'alpha': 1}, ax=ax[1])

ax[1].set(xlabel='Uniform Distribution ', ylabel='Frequency')

plt.tight_layout()
plt.show()
