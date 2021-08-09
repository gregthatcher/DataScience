'''
Ideas from : https://app.pluralsight.com/course-player?clipId=eaf84c38-4882-42a6-923a-cee4a2d3fa91
Using pre-processed car data, predict the price of cars
TODO: Why are my results (R^2, root mean square error) better than results in
the video above?
Note that regression models are considered to be a High Bias Algorithm
(as opposed to Decision Trees and Dense Neural Networks, which are
high Variance Algorithms)
In high Variance algorithms, parameters are complex
In high Bias algorithms, parameters are simple but
assumptions about model are great (e.g. "model is linear").

TODO: Note that in addition to Lasso and Ridge (I have examples below),
we also hae Elastic Net Regression which combines both Lasso and Ridge.
'''

import math
from matplotlib.colors import Normalize
import pandas as pd
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def analyze_model(model, model_name, X_train, y_train, X_test, y_test):
    # copy_X = True means data will be copied and not modified
    # fit_intercept means data is not "centered"
    # normalize=False means data won't be "normalized" (set between 0 and 1;
    # subtract mean, divide by L2 norm); note that "standardization"
    # is probably a better option here
    # Also, the L2 norm is the Euclidean Distance
    # L1 norm is the Manhatten distance
    # See https://machinelearningmastery.com/vector-norms-machine-learning/
    print(f"\nAnalysis of {model_name} model...")
    print(model.get_params())

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    # How well do we score on training data
    print("Train Data Score (r^2; how much of the data's variance are we"
          " capturing):")
    print(train_score)
    print("Test Data Score (r^2; how much of the data's variance are we "
          "capturing):")
    print(test_score)

    print("Features sorted by importance")
    predictors = X_train.columns
    # Here, we create a Panda Series using the model coefficients
    # as the data, and the column names as the index
    coef = pd.Series(model.coef_, predictors).sort_values()
    print(coef)

    # Let's make a prediction
    y_predict = model.predict(X_test)
    # print(y_predict)

    # Root mean square error (standard deviation of errors)
    # tells us how much, on average our prediction will deviate
    # from actual
    root_mean_square_error = math.sqrt(mean_squared_error(y_predict, y_test))

    plt.figure(figsize=(15, 6))
    plt.plot(y_predict, label="Predicted")
    plt.plot(y_test.values, label="Actual")
    plt.text(
        12, 30000, "Expected prediction deviation from true value: "
        f"{root_mean_square_error:.2f} dollars")
    plt.text(
        0, 35000, f"Test Score {test_score:.2f}\nTrain Score "
        f"{train_score:.2f}")
    plt.ylabel("Price")
    plt.legend()
    plt.title(model_name)
    plt.show()


# Data was prepared by
# data_preparation/automobile_price_prediction_data_prep.py
auto_data = pd.read_csv("data/preprocessed_data/car_prices.csv")

# print(auto_data.head())
# print(auto_data.tail())

X = auto_data.drop("price", axis=1)

y = auto_data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=11)

# Note that adding normalize=True makes model go 
# crazy, why?  It works below for Lasso??
# linear_model = LinearRegression(normalize=True)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
analyze_model(linear_model, "Linear Regression",
              X_train, y_train, X_test, y_test)

# Lasso ~ Least Absolute Shrinkage and Selection Operator
#  Alpha is regularization parameter
# "Regularization" is one way to combat
# complex (high variance) models.
# Regularization reduces variance error, but
# increases Bias error.
# Alpha is multipled by L1 norm (sum of absolute values) and
# added to objective function which previously only
# tried to minimize least squares (sqrt((y_actual-y_predicted)^2))
# so we end up with
# (sqrt((y_actual-y_predicted)^2))
# + Alpha * (|First Coefficient|^1 + [Second Coefficient]^1 + ...)
# That is, we add the L1 norm of the coefficients as a penalty.
#
# By normalizing, we are "centering" by subtracting the mean
# and dividing by L2 norm
#
# Note that Lasso regression will reduce some
# coefficients to zero;  we can remove these
# features if we're sure they weren't colinear with
# other features.  This is one way to find
# out which features are important or not.
# By tuning Alpha, we can elimate unimportant features.
lasso_model = Lasso(alpha=0.5, normalize=True)
lasso_model.fit(X_train, y_train)
analyze_model(lasso_model, "Lasso - Alpha=0.5",
              X_train, y_train, X_test, y_test)
# Now, let's change alpha parameter
# A higher alpha causes many more features
# to have zero as coefficient
lasso_model = Lasso(alpha=5, normalize=True)
lasso_model.fit(X_train, y_train)
analyze_model(lasso_model, "Lasso - Alpha = 5",
              X_train, y_train, X_test, y_test)

# For Ridge Regression, we modify the objective function
# as follows:
# (sqrt((y_actual-y_predicted)^2))
# + Alpha * (|First Coefficieng|^2 + [Second Coefficient]^2 + ...)
# That is, we add the L2 norm of the coefficients as a penalty.
#
# Unlike Lasso, tuning the alpha parameter will not
# force (unimportant feature) coefficients to zero, so its
# not suitable for finding unimportant (linear) features.
#
# The absolute value in the L1 norm makes it impossible
# to find a "closed form solution".  This means that
# Lasso tries many models to find a solution.  In contrast,
# Ridge Regression _does_ have a closed form solution
# (it can be expressed/calculated as a formula),
# so it only tries one model.
# TODO: Let's try some "hyperparamter tuning" to find best Alpha
# Note that a high value (1.0) causes test score to fall
for alpha in [0.05, 0.5, 1.0]:
    ridge_model = Ridge(alpha=alpha, normalize=True)
    ridge_model.fit(X_train, y_train)
    analyze_model(ridge_model, f"Ridge - Alpha = {alpha}",
                  X_train, y_train, X_test, y_test)

# SVM _Classification_ tries to use "support vectors"
# to create a maximum margin around the separating
# hyperplane with no data points within the margin.
# (i.e. with no points inside the margin)
# In contrast, SVM Regression tries to create a margin
# around a hyerplane which has as many points as possible
# _inside_ the margin.  The width of this margin
# is controlled by the hyperparameter, "epsilon".
# (the total width of the margin around the hyperplane is 2 * epsilon)
# (in contrast, in SVM Classification, the width of the margin is found
# by the optimizer).
# In summary, SVM Classification has an objective
# function with big penalties for points on the wrong side
# of the margin, whereas SVM Regression has penalties
# for points which are outside the margin -- it doesn't
# matter if these points are on the correct side or not.
# These penalized "Margin Violations" are multiplied by
# the hyperparameter "C".
