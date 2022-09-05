# Idea from : https://campus.datacamp.com/courses/statistical-thinking-in-python-part-2/hypothesis-test-examples?ex=2
# Did party affiliation make a difference in the Civil Rights act of 1964
# In general, in an A/B test, the null hypothesis is that the
# test statistic is impervious to the change.

import numpy as np
from null_hypthosis_significance_testing_with_permutation_samples \
    import draw_perm_reps


dems = np.array([True] * 153 + [False] * 91)
reps = np.array([True] * 136 + [False] * 35)


def frac_yea_dems(dems, reps):
    """Compute fraction of Democrat yea votes."""
    frac = np.sum(dems) / len(dems)
    return frac


# Acquire permutation samples: perm_replicates
perm_replicates = draw_perm_reps(dems, reps, frac_yea_dems, size=10000)

# Compute and print p-value: p
p = np.sum(perm_replicates <= 153/244) / len(perm_replicates)
print('p-value =', p)
