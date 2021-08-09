
'''
Clean up data for "automobile price prediction" from
https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data
Ideas from: https://app.pluralsight.com/course-player?clipId=eaf84c38-4882-42a6-923a-cee4a2d3fa91
'''

import os
import pandas as pd
import numpy as np

print(os.getcwd())
auto_data = pd.read_csv("data/raw_data/imports-85.data",
                        sep=r'\s*,\s*', engine="python",
                        header=None,
                        names=[
                            "symboling",  # -3, -2, -1, 0, 1, 2, 3."
                            "normalized-losses",  # continuous from 65 to 256.
                            "make",  # alfa-romero, audi, bmw, chevrolet,
                            # dodge, honda,
                            # isuzu, jaguar, mazda, mercedes-benz, mercury,
                            # mitsubishi, nissan, peugot, plymouth, porsche,
                            # renault, saab, subaru, toyota, volkswagen, volvo
                            "fuel-type",  # diesel, gas.
                            "aspiration",  # std, turbo.
                            "num-of-doors",  # four, two.

                            "body-style",  # hardtop, wagon, sedan, hatchback,
                            # convertible.
                            "drive-wheels",  # 4wd, fwd, rwd.
                            "engine-location",  # front, rear.
                            "wheel-base",  # continuous from 86.6 120.9.
                            "length",  # continuous from 141.1 to 208.1.
                            "width",  # continuous from 60.3 to 72.3.
                            "height",  # continuous from 47.8 to 59.8.
                            "curb-weight",  # continuous from 1488 to 4066.
                            "engine-type",  # dohc, dohcv, l, ohc, ohcf, ohcv,
                                            # rotor.
                            "num-of-cylinders",  # eight, five, four, six,
                                                # three, twelve, two.
                            "engine-size",  # continuous from 61 to 326.

                            "fuel-system",  # 1bbl, 2bbl, 4bbl, idi, mfi,
                                            #  mpfi, spdi, spfi.
                            "bore",  # continuous from 2.54 to 3.94.
                            "stroke",  # continuous from 2.07 to 4.17.
                            "compression-ratio",  # continuous from 7 to 23.
                            "horsepower",  # continuous from 48 to 288.
                            "peak-rpm",   # continuous from 4150 to 6600.
                            "city-mpg",  # continuous from 13 to 49.
                            "highway-mpg",  # continuous from 16 to 54.
                            "price"  # continuous from 5118 to 45400.
                        ])

print("Shape ", auto_data.shape)
print(auto_data.head())
print(auto_data.tail())

# Missing values are currently "?"
auto_data = auto_data.replace("?", np.nan)
print(auto_data.head())

print(auto_data.describe())
# Let's see "object" types also
print(auto_data.describe(include="all"))

for column in ["price", "horsepower"]:
    print(f"Before changing {column}:", auto_data[column].describe(), sep="\n")
    auto_data[column] = pd.to_numeric(auto_data[column], errors="coerce")
    print(f"\nAfter changing {column}", auto_data[column].describe(), sep="\n")

# Get rid of columns we know aren't predictors
# "normalized_losses" have something to do with insurance
auto_data = auto_data.drop("normalized-losses", axis=1)
print("After dropped column", auto_data.describe(), sep="\n")

cylinders_dict = {"two": 2,
                  "three": 3,
                  "four": 4,
                  "five": 5,
                  "six": 6,
                  "eight": 8,
                  "twelve": 12
                  }

auto_data["num-of-cylinders"].replace(cylinders_dict, inplace=True)
print("After mapping num-of-cylinders",
      auto_data["num-of-cylinders"].describe(), sep="\n")

# Convert categorical into one-hot
auto_data = pd.get_dummies(auto_data,
                           columns=["make",
                                    "fuel-type",
                                    "aspiration",
                                    "num-of-doors",
                                    "body-style",
                                    "drive-wheels",
                                    "engine-location",
                                    "engine-type",
                                    "fuel-system"])
print("After One-Hot Encoding", auto_data.head(), sep="\n")

# Drop all NA values
print("Before dropna:", auto_data.shape)
auto_data = auto_data.dropna()
print("After dropna:", auto_data.shape)

print("Any Nulls", auto_data[auto_data.isnull().any(axis=1)])

print("Columns that are object type",
      list(auto_data.select_dtypes(['object']).columns))

print(auto_data[["bore", "stroke", "peak-rpm"]].head())

auto_data[["bore", "stroke", "peak-rpm"]
          ] = auto_data[["bore", "stroke", "peak-rpm"]].apply(pd.to_numeric)
print("Columns that are object type",
      list(auto_data.select_dtypes(['object']).columns))
auto_data.info()
auto_data.to_csv("data/preprocessed_data/car_prices.csv", index=False)
