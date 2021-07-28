'''
Ideas from : https://app.pluralsight.com/course-player?clipId=eaf84c38-4882-42a6-923a-cee4a2d3fa91
Using pre-processed car data, predict the price of cars
TODO: Why are my results (R^2, root mean square error) better than results in
the video above?
Note that regression models are considered to be a High Bias Algorithm
(as opposed to Decision Trees and Dense Neural Networks, which are
high Variance Algorithms)
'''

import math
import pandas as pd
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

auto_data = pd.read_csv("data/preprocessed_data/car_prices.csv")

# print(auto_data.head())
# print(auto_data.tail())

X = auto_data.drop("price", axis=1)

y = auto_data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=11)


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# copy_X = True means data will be copied and not modified
# fit_intercept means data is not "centered"
# normalize=False means data won't be "normalized" (set between 0 and 1;
# subtract mean, divide by L2 norm); note that "standardization"
# is probably a better option here
# Also, the L2 norm is the Euclidean Distance
# L1 norm is the Manhatten distance
# See https://machinelearningmastery.com/vector-norms-machine-learning/
print(linear_model.get_params())

# How well do we score on training data
print("Train Data Score (r^2; how much of the data's variance are we capturing):")
print(linear_model.score(X_train, y_train))
print("Test Data Score (r^2; how much of the data's variance are we capturing):")
print(linear_model.score(X_test, y_test))

print("Features sorted by importance")
predictors = X_train.columns
# Here, we create a Panda Series using the model coefficients
# as the data, and the column names as the index
coef = pd.Series(linear_model.coef_, predictors).sort_values()
print(coef)


# Let's make a prediction
y_predict = linear_model.predict(X_test)
print(y_predict)

# Root mean square error (standard deviation of errors)
# tells us how much, on average our prediction will deviate
# from actual
root_mean_square_error = math.sqrt(mean_squared_error(y_predict, y_test))

plt.figure(figsize=(15, 6))
plt.plot(y_predict, label="Predicted")
plt.plot(y_test.values, label="Actual")
plt.text(12, 30000, f"Expected prediction deviation from true value:{root_mean_square_error:.2f}")
plt.ylabel("Price")
plt.legend()
plt.show()
