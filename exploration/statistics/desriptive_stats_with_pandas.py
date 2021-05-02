'''
This module contains routines for doing descriptive statistics (not inferential) using pandas dataframes
'''
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

SEPAL_LENGTH = "sepal_length"
SEPAL_WIDTH = "sepal_width"
PETAL_LENGTH="petal_length"
PETAL_WIDTH="petal_width"

iris = sns.load_dataset('iris')
print(type(iris))

print(pd.__version__)

print(iris.head())
print(iris.isnull().sum())

print(iris.isnull().head())

print(iris.species.value_counts())

print("\nMinimums:")
print(iris.min())
print("\nMaximums:")
print(iris.max())
print("\nMeans:")
print(iris.mean())

print("\nRanges:")
print("Sepal Length:", round(max(iris[SEPAL_LENGTH]) - min(iris[SEPAL_LENGTH]), 2))
print("Sepal Width:", round(max(iris[SEPAL_WIDTH]) - min(iris[SEPAL_WIDTH]), 2))
print("Petal Length:", round(max(iris[PETAL_LENGTH]) - min(iris[PETAL_LENGTH]), 2))
print("Petal Width:", round(max(iris[PETAL_WIDTH]) - min(iris[PETAL_WIDTH]), 2))

species_names = ["setosa", "virginica", "versicolor"]
titles = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
column_names = [SEPAL_LENGTH, SEPAL_WIDTH, PETAL_LENGTH, PETAL_WIDTH]

fig, ax = plt.subplots(1+len(species_names), len(titles), figsize=(14, 10))

# Show histograms of each column
for column, (title, column_name) in enumerate(zip(titles, column_names)): 
    ax[0][column].hist(iris[column_name])
    ax[0][column].set_title(title)
    ax[0][column].axvline(iris[column_name].median(), color="g", label="median", linewidth=5)
    ax[0][column].axvline(iris[column_name].mean(), color="r", label="mean")
    modes = iris[column_name].mode()
    for m in modes:
        ax[0][column].axvline(m, color="y", label="mode", linestyle="--")
    ax[0][column].legend()

for row, species_name in enumerate(species_names, start=1):
    # Show histograms of each column for each species_name (e.g. "setosa")
    for column, (title, column_name) in enumerate(zip(titles, column_names)): 
        data = iris[iris["species"] == species_name]
        ax[row][column].hist(data[column_name])
        ax[row][column].set_title(f"{species_name.capitalize()}-{title}")
        ax[row][column].axvline(data[column_name].median(), color="g", label="median", linewidth=5)
        ax[row][column].axvline(data[column_name].mean(), color="r", label="mean")
        modes = data[column_name].mode()
        for m in modes:
            ax[row][column].axvline(m, color="y", label="mode", linestyle="--")

        ax[row][column].legend()

fig.tight_layout()
plt.show()


