'''
TODO: Mean Shift Clustering (after standardization??) might be something
we want to add to data during "Feature Engineering"
From https://www.geeksforgeeks.org/ml-mean-shift-clustering/
"Given a set of data points, the algorithm iteratively assigns each data point
towards the closest cluster centroid and direction to the closest cluster
centroid is determined by where most of the points nearby are at. So each
iteration each data point will move closer to where the most points are at,
which is or will lead to the cluster center. When the algorithm stops, each
point is assigned to a cluster."

"Unlike the popular K-Means cluster algorithm, mean-shift does not require
specifying the number of clusters in advance. The number of clusters is
determined by the algorithm with respect to the data.

The downside to Mean Shift is that it is computationally expensive O(n²)."

The algorithm uses Kernel Density Estimation (KDE), a method that estimates
the underlying distribution (probability density) by placing a "kernel"
(weighting function, generally used in convolution) on each point in the
data set.  The simplest kernel is just a circle around the given point
that counts the number of points inside it.  More generally, a Gaussian kernel
is used.

Instead of setting the number of clusters as a hyperparameter (as we do with
K Means Clustering), we set a "bandwidth" parameter, specifying how
wide an area to check around each point.  Really, the bandwidth parameter
is proportional to the standard deviation of the gaussian (rbf) kernel,
but we can think of it as the radius of a circle around our given point.

We then do a density calculation for all points within the "bandwidth" of
the given point to find the most dense location.  This gives us a vector
direction which will move our point towards the "mode" of that given area.
In this way, each point "moves" towards its final cluster.  When the points
stop moving (much), we have our clusters.

In contrast to K-Means Clustering
1.) K-Means can't handle some non-linear data (Kernel function takes care of
this for Mean-Shift Clusturing).
2.) It's harder to tune bandwidth parameter than it is to tune # of clusters
3.) Mean Shift Clustering is much more computationally expensive O(n^2) vs
    O(n))
4.) K-Means struggles with outliers; not so with Mean Shift Clustering
'''

import collections
import numpy as np
import pandas as pd
from scipy.sparse import data
from sklearn.cluster import MeanShift, estimate_bandwidth

titanic_data = pd.read_csv("data/preprocessed_data/titanic_train.csv")

print("Shape: ", titanic_data.shape)
print("Columns: ", titanic_data.columns)
print(titanic_data.head())

# Let's estimate the best bandwidth parameter.
# Note that this runs in quadratic (N^2) time.
# The bandwidth parameter sets the "shape of the kernel" -
# smaller values give tall skinny kernels, larger values
# give short fat kernels.
# For a Gaussian (RBF) Kernel (the default) the bandwidth
# is proportional to the standard deviation of the kernel.
# Note that bandwidth is the _only_ hyperparamter for
# Mean Shift Clustering
best_bandwidth = estimate_bandwidth(titanic_data)
print("Best Bandwidth: ", best_bandwidth)

# analyzer = MeanShift(bandwidth=50)
analyzer = MeanShift(bandwidth=best_bandwidth)

# TODO: Why didn't we standardize/normalize before this?
analyzer.fit(titanic_data)

# Let's see how many clusters were created
# this returns a "numpy.ndarray"
labels = analyzer.labels_

print("Unique Labels: ", np.unique(labels), type(labels))

# Commenting out method shown in class, as one-liner
# (titanic_data["cluster_group"] = labels) seems better
# titanic_data["cluster_group2"] = np.nan
# data_length = len(titanic_data)
# for i in range(data_length):
#     titanic_data.iloc[i, titanic_data.columns.get_loc(
#         "cluster_group2")] = labels[i]

# Here is better method
titanic_data["cluster_group"] = labels

# print(any(titanic_data["cluster_group"] != titanic_data["cluster_group2"]))


print(titanic_data.head())

# Around 40% of people survived
# and more people in lower class cabins
# We have more males than females (male=1; female=0)
# Average fare around $35
print(titanic_data.describe())

titanic_cluster_data = titanic_data.groupby(["cluster_group"]).mean()

print(titanic_cluster_data)

# Let's add count data to our "cluster_group"
titanic_cluster_data["Counts"] = titanic_data.groupby(["cluster_group"]).size()
# This returns a Pandas Series
# print(type(titanic_data.groupby(["cluster_group"]).size()))
# This returns a Pandas DataFrame
# print(type(titanic_data.groupby(["cluster_group"]).mean()))

print(titanic_cluster_data)
