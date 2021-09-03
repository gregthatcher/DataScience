'''
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
3.) Mean Shift Clustering is much more computationally expensive O(n^2) vs O(n))
4.) K-Means struggles with outliers; not so with Mean Shift
'''

import pandas as pd

titanic_data = pd.read_csv("data/raw_data/Titanic/train.csv", quotechar='"')

print(titanic_data.columns)
print(titanic_data.head())