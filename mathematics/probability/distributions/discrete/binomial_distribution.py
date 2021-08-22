'''
From https://www.unf.edu/~cwinton/html/cop4300/s09/class.notes/DiscreteDist.pdf
"The Binomial Distribution represents the number of successes and failures in
n independent Bernoulli trials for some given value of n."

'''


from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

# n is number of trials
# p is probability of success in one trial
# size is number of data points we want
sns.distplot(random.binomial(n=10, p=0.5, size=10000), hist=True, kde=False)
plt.title("p=0.5;n=10;s=10000")
plt.show()

sns.distplot(random.binomial(n=10, p=0.1, size=10000), hist=True, kde=False)
plt.title("p=0.1;n=10;s=10000")
plt.show()
