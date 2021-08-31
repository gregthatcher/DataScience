'''
Predicting car prices using Gradient Boosting
Ideas from https://app.pluralsight.com/course-player?clipId=85cca391-18ce-460f-8b9a-f189c10b9d2f
'''


import math
from numpy.core.fromnumeric import mean
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

'''
Gradient Boosting
a form of Ensemble (French for "together") Learning
which can be thought of as another regularization technique
to avoid over-fitting (or "variance error" - when model changes
drastically with different training data).

Pretend we are using Linear Models for weak learners:
(note that OLS "weak learners" are not good for Gradient Boosting,
we are just using this to illustrate a point).

Model 1 ("Weak Learner"):
    y = A1 + B1*x + e1
    (e1 is residual error - what we need to make model correct)

Model 2 ("Weak Learner"):
    e1 = A2 + B2 * x + e2 (e2 is residuals from Model 2)

Model 3 ("Weak Learner"):
    e2 = A3 + B3 * x + e3

Combined Model ("Strong Learner"):
    y = A1 + A2 + A3 + (B1 + B2 + B3) * x + e3
    (Note that e3 is what our final model failed to learn)

Typically, we use 100-200 (or more) weak models

Gradient Boosting does not work well with Regression models
    Examples:
    e.g. OLS (Ordinary Least Squares) using MSE (Mean Squared Error; used only
    for normal data??)

Gradient boosting does work well with Decision Trees
Note that Decision Trees have only one hyperparameter, the depth.
By setting the depth to a low number, we prevent overfitting in
all the "weak" learners.

Gradient Boosting also has a "Shrinkage Factor".  This factor
is multipled by outputs from each weak learner, thus slowing down
learning and mitigating overfitting.
If you have more learners, shrink a lot.
If you have a few learners, shrink a little.
Shrinkage Factors are typically beteen 0.1 and 1.
'''


def show_model_results(gbr_model):
    # Video gives 99% accuracy, but I only get 89%.  Why??
    # Answer, she was using test data, not train
    # print("Test R Squared", gbr_model.score(X_train, y_train))
    score = gbr_model.score(X_test, y_test)
    print(f"Score {score}")

    y_predict = gbr_model.predict(X_test)
    r_square = gbr_model.score(X_test, y_test)
    mse = math.sqrt(mean_squared_error(y_predict, y_test))

    plt.figure(figsize=(15, 6))

    plt.plot(y_predict, label="Predicted")
    plt.plot(y_test.values, label="Actual")
    plt.ylabel("Price")
    plt.text(15, 30000, f"R Squared: {r_square:0.2f}\n\n"
                        f"Expected variance from prediction: {mse:0.2f}")
    plt.legend()
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

params = {"n_estimators": 500,  # number of "weak learners"/"boosting stages"
          "max_depth": 6,  # maximum depth of each decision tree
          "min_samples_split": 2,  # node must have 2 samples before we split
          "learning_rate": 0.01,  # "Shrinkage Factor"
          "loss": "ls"}  # use least square regression loss function
gbr_model = GradientBoostingRegressor(**params)
gbr_model.fit(X_train, y_train)

show_model_results(gbr_model)

# Now, let's find best hyper paramters
num_estimators = [100, 200, 500]
learn_rates = [0.01, 0.02, 0.05, 0.1]
max_depths = [4, 6, 8]

param_grid = {"n_estimators": num_estimators,
              "learning_rate": learn_rates,
              "max_depth": max_depths}
grid_search = GridSearchCV(GradientBoostingRegressor(min_samples_split=2,
                                                     loss="ls"),
                           param_grid, cv=3, return_train_score=True)
grid_search.fit(X_train, y_train)
print("Best Parameters: ", grid_search.best_params_)


print("\nAll Grid Search Results:")
for i in range(36):
    print("Parameters: ", grid_search.cv_results_["params"][i])
    print("Mean Test Score: ", grid_search.cv_results_["mean_test_score"][i])
    print("Rank: ", grid_search.cv_results_["rank_test_score"][i])
    print()

gbr_model = GradientBoostingRegressor(
    n_estimators=100,  # number of "weak learners"/"boosting stages"
    max_depth=4,  # maximum depth of each decision tree
    min_samples_split=2,  # node must have 2 samples before we split
    learning_rate=0.1,  # "Shrinkage Factor"
    loss="ls")  # use least square regression loss function

gbr_model.fit(X_train, y_train)

show_model_results(gbr_model)
