
'''
Clean up data for "automobile mileage prediction" from
https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data
Ideas from: https://app.pluralsight.com/course-player?clipId=5deae079-7e1d-4226-a653-525ba16c9769
'''
import numpy as np
import pandas as pd

auto_data = pd.read_csv("data/raw_data/AutoMileage/auto-mpg.data",
                        delim_whitespace=True,
                        header=None,
                        names=["mpg",
                               "cylinders",
                               "displacement",
                               "horsepower",
                               "weight",
                               "acceleration",
                               "model",
                               "origin",
                               "car_name"])

print(auto_data.head())

# Let's see if car name helps us in any way
shape = auto_data.shape
print(f"Shape of Data {shape}")
num_unique_car_names = len(auto_data["car_name"].unique())
print(f"Number of unique car names {num_unique_car_names}")
# So, car_name has 305 unique values out of 398 -- not useful
auto_data.drop("car_name", axis=1, inplace=True)
print(auto_data.head())

# The origin column (a number) tells us what country the car
# is from.  A better representation would be one-hot encoding
auto_data["origin"] = auto_data["origin"].replace(
    {1: "america", 2: "europe", 3: "asia"})
auto_data = pd.get_dummies(auto_data, columns=["origin"])
print(auto_data.head())
auto_data = auto_data.replace("?", np.nan)
print("Number of nulls", auto_data.isnull().sum().sum())
auto_data.dropna(inplace=True)
print("Number of nulls", auto_data.isnull().sum().sum())

print("Columns that are object type",
      list(auto_data.select_dtypes(['object']).columns))

# Horsepower is an object
print("Horse Power Sample", auto_data.horsepower.head())

auto_data["horsepower"] = pd.to_numeric(auto_data["horsepower"])

print("Columns that are object type",
      list(auto_data.select_dtypes(['object']).columns))

print("Horse Power Sample", auto_data.horsepower.head())

auto_data.info()

auto_data.to_csv("data/preprocessed_data/car_mileage.csv", index=False)

