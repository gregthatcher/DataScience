'''
Ideas from : https://app.pluralsight.com/course-player?clipId=5deae079-7e1d-4226-a653-525ba16c9769
Using pre-processed car data, predict the mileage of cars
'''

import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

auto_data = pd.read_csv("data/preprocessed_data/car_mileage.csv")

X = auto_data.drop("mpg", axis=1)
y = auto_data["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=11)


def analyze_model(X_train, y_train, X_test, y_test, kernel, C):
    regression_model = SVR(kernel=kernel, C=C)
    regression_model.fit(X_train, y_train)

    train_score = regression_model.score(X_train, y_train)
    test_score = regression_model.score(X_test, y_test)

    y_predict = regression_model.predict(X_test)
    root_mean_square_error = math.sqrt(mean_squared_error(y_predict, y_test))

    plt.figure(figsize=(15, 6))
    plt.plot(y_predict, label="Predicted")
    plt.plot(y_test.values, label="Actual")
    plt.text(
        12, 40, "Expected prediction deviation from true value:"
        f"{root_mean_square_error:.2f} mpg")
    plt.text(
        0, 40, f"Test Score {test_score:.2f}\nTrain Score "
        f"{train_score:.2f}")
    plt.ylabel("Mileage")
    plt.legend()
    plt.title(f"SVR - kernel={kernel} C={C}")
    plt.show()

    # Apparently, we only have coef_ when using a linear kernel
    if kernel == "linear":
        # Let's see what coefficients look like
        predictors = X_train.columns
        coef = pd.Series(regression_model.coef_[0], predictors).sort_values()
        coef.plot(kind="bar", title=f"Model Coefficients - {kernel}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# Not trying "precomputed", as it needs more research
for kernel in ["linear", "rbf", "poly", "sigmoid"]:
    # "C" is the penalty factor for points that lie outside
    # our margins
    # Note that the default for epsilon (width of each margin)
    # is 0.1
    analyze_model(X_train, y_train, X_test, y_test, kernel, 1.0)

# Reducing the C parameter reduces the penalty, so the model
# focuses more on fitting to the data rather than penalizing
# points outside the margin; note that C=0 means no penalty,
# but model throws ValueError exception for this.
for C in [1.0, 0.5, 0.25, 0.1, 0.01]:
    analyze_model(X_train, y_train, X_test, y_test, "linear", C)
