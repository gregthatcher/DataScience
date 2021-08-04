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
assumptions about model are great (e.g. "model is linear")
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
        12, 30000, "Expected prediction deviation from true value:"
        f"{root_mean_square_error:.2f}")
    plt.text(
        0, 35000, f"Test Score {test_score:.2f}\nTrain Score "
        f"{train_score:.2f}")
    plt.ylabel("Price")
    plt.legend()
    plt.title(model_name)
    plt.show()


auto_data = pd.read_csv("data/preprocessed_data/car_prices.csv")

# print(auto_data.head())
# print(auto_data.tail())

X = auto_data.drop("price", axis=1)

y = auto_data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=11)


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
analyze_model(linear_model, "Linear Regression",
              X_train, y_train, X_test, y_test)

# Alpha is regularization parameter
# This is multipled by L1 norm (sum of absolute values)
# by normalizing, we are "centering" by subtracting the mean
# and dividing by L2 norm
# Note that Lasso regression will reduce some
# coefficients to zero;  we can remove these
# features if we're sure they weren't colinear with
# other features.
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
