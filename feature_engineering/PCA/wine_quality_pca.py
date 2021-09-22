'''

Principal Components Analysis (PCA)

Ideas from:
https://app.pluralsight.com/course-player?clipId=74e0c01c-2bc8-4605-b88a-80b8e830a2be
https://app.pluralsight.com/course-player?clipId=c84c31ec-c392-48b8-b698-19c383560bf0

The goal of PCA is to re-express data in terms of a few, 
well-chosen vectors (Principal Components), so that we can
reduce the dimensionality of our data while maintaining (most)
of the variation in our data.
We do this by choosing the vectors that capture the most variation
in the data.  That is, the projections onto each vector
should maximize the distance between (projected) points.
The first vector is chosen to find the maximum variation,
and subsequent vectors are chosen to be orthogonal to previous
vectors, and also give maximum variance.  Since the vectors
are orthogonal, we know we are minimizing the number of vectors.
Our goal is to discard some of the principal components,
so that we reduce the dimensionality of our data.
Note that in general, there are as many principal components
as there are dimensions in the original data.

Note that this example "scales" our data.
Does this answer the question of should
we normalize or standardize before PCA??
'''

import pandas as pd
# So we can use "Standard Scaler"
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns


def calc_pca(num_vectors, X, Y):
    pca = PCA(n_components=num_vectors, whiten=True)
    X_reduced = pca.fit_transform(X)
    X_train, x_test, Y_train, y_test = train_test_split(
        X_reduced, Y, test_size=0.2, random_state=0)
    return X_train, x_test, Y_train, y_test


def calc_svc_accuracy(X_train, Y_train, x_test, y_test):
    clf_svc = LinearSVC(penalty="l1", dual=False, tol=1e-3)
    clf_svc.fit(X_train, Y_train)
    accuracy = clf_svc.score(x_test, y_test)
    return accuracy


# Note that this dataset is already "clean" (without missing values)
# and has only numeric fields
wine_data = pd.read_csv("data/raw_data/Wine-Quality/winequality-white.csv",
                        names=[
                            'Fixed Acidity',
                            "Volatile Acidity",
                            "Citric Acid",
                            "Residual Sugar",
                            "Chlorides",
                            "Free Sulfur Dioxide",
                            "Total Sulfur Dioxide",
                            "Density",
                            "pH",
                            "Sulphates",
                            "Alcohol",
                            "Quality"
                        ],
                        skiprows=1,
                        sep=r'\s*;\s*',
                        engine="python")

print(wine_data.head())
# Note that all fields are numeric (clean)
wine_data.info()

# We have seven values for the target
# if we guessed, our accuracy would be 1/7 (14%)
quality_values = sorted(wine_data["Quality"].unique())
# quality_values.sort()
print("Unique Values for Quality", quality_values)

X = wine_data.drop("Quality", axis=1)
Y = wine_data["Quality"]

# Standardize the data
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html
# "Center to the mean and component wise scale to unit variance."
# This is done by subtracting mean, and dividing by standard deviation
X = preprocessing.scale(X)
X_train, x_test, Y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# Let's get a baseline score for our model
# Random guesses would give us about 14% accuracy (1/7)
# We get almost 50%, so this is good
accuracy = calc_svc_accuracy(X_train, Y_train, x_test, y_test)
print("Accuracy before PCA: ", accuracy)

# Dimensionality Reduction only works if features are correlated with each
# other.  Let's see...
corr_matrix = wine_data.corr()
f, ax = plt.subplots(figsize=(7, 7))
sns.set(font_scale=0.9)
sns.heatmap(corr_matrix, vmax=.8, square=True,
            annot=True, fmt='.2f', cmap="winter")
# We see a number of correlations, including Density vs. Residual Sugar
# and Sulfur Dioxide vs. Free Sulfur Dioxide
plt.show()

# We'll start with 11 features, as that is what our dataset has
# (We don't expect this to increase accuracy)
pca = PCA(n_components=11, whiten=True)
X_reduced = pca.fit_transform(X)

# Note that we're using explained_variance_ratio
# and not just explanied_variance so we get
# "percentage terms"
print(pca.explained_variance_ratio_)

# This plot is called a "scree plot"
# This plot helps us determine which features
# are important
# Note that the "elbow" here shows that
# most variance is captured by two dimensions
plt.plot(pca.explained_variance_ratio_)
plt.xlabel("Dimension")
plt.ylabel("Explained Variance Ratio")
plt.show()

# Let's try to train model again...
# Note that accuracy is _exactly_ the same
# with transformed dimensions 
X_train, x_test, Y_train, y_test = train_test_split(
    X_reduced, Y, test_size=0.2, random_state=0)
accuracy = calc_svc_accuracy(X_train, Y_train, x_test, y_test)
print("Accuracy after PCA: ", accuracy)


# Let's try getting rid of last two dimenensions
# as these are least relevant
X_train, x_test, Y_train, y_test = calc_pca(9, X, Y)
accuracy = calc_svc_accuracy(X_train, Y_train, x_test, y_test)
print("Accuracy after reducing # dimensions by two: ", accuracy)

X_train, x_test, Y_train, y_test = calc_pca(5, X, Y)
accuracy = calc_svc_accuracy(X_train, Y_train, x_test, y_test)
print("Accuracy after reducing # dimensions by five: ", accuracy)
