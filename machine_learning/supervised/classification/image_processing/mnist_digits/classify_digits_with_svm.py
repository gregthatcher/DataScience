'''
Classify MNIST digits using SVM which supports multi-class classification
'''

import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

mnist_data = pd.read_csv("./data/raw_data/MNIST2/train.csv")

print(mnist_data.head())

features = mnist_data.columns[1:]
X = mnist_data[features]
y = mnist_data["label"]

print("Type of data: ", type(X))
print("Type of label: ", type(y))

# We divide by 255 to get intensities between 0 and 1
X_train, X_test, y_train, y_test = train_test_split(
    X/255., y, test_size=0.1, random_state=0)

# Instantiate Linear Support Vector Classifier
clf_svm = LinearSVC(penalty="l2", dual=False, tol=1e-5)
clf_svm.fit(X_train, y_train)

predictions = clf_svm.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy {accuracy}")

# Now, let's do Grid Search to find best parameters
penalties = ["l1", "l2"]
tolerances = [1e-3, 1e-4, 1e-5]
param_grid = {"penalty": penalties, "tol": tolerances}
# cv=3 divides our data into 3 parts, with 2 parts being used for training,
# and 1 part being used for validation
grid_search = GridSearchCV(LinearSVC(dual=False), param_grid, cv=3)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

# Now, let's try the best parameters
clf_svm = LinearSVC(penalty="l1", dual=False, tol=1e-4)
clf_svm.fit(X_train, y_train)

predictions = clf_svm.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy (for 'best' parameters) {accuracy}")
