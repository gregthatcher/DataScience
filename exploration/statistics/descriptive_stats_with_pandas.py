'''
This module contains routines for doing descriptive statistics (not inferential) using pandas dataframes
'''
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

SEPAL_LENGTH = "sepal_length"
SEPAL_WIDTH = "sepal_width"
PETAL_LENGTH = "petal_length"
PETAL_WIDTH = "petal_width"


def print_descriptive_stats(iris, species):
    print("Type: ", type(iris))

    print("Pandas Version: ", pd.__version__)

    print("\nShape of DataFrame:", iris.shape)
    print("\nFirst Few Rows:")
    print(iris.head())
    print("\nNull Counts:")
    print(iris.isnull().sum())

    print("\nValue Counts:")
    print(iris.species.value_counts())

    print("\nMinimums:")
    print(iris.min())
    print("\nMaximums:")
    print(iris.max())
    print("\nMeans:")
    print(iris.mean())
    print("\nVariances:")
    print(iris.var())
    print("\nStandard Deviations:")
    print(iris.std())

    print("\nRanges:")
    print("Sepal Length:", round(
        max(iris[SEPAL_LENGTH]) - min(iris[SEPAL_LENGTH]), 2))
    print("Sepal Width:", round(
        max(iris[SEPAL_WIDTH]) - min(iris[SEPAL_WIDTH]), 2))
    print("Petal Length:", round(
        max(iris[PETAL_LENGTH]) - min(iris[PETAL_LENGTH]), 2))
    print("Petal Width:", round(
        max(iris[PETAL_WIDTH]) - min(iris[PETAL_WIDTH]), 2))

    print("\nInterquartile Ranges (50% of data is in this range):")
    print("Sepal Length:", round(iris[SEPAL_LENGTH].quantile(
        0.75) - iris[SEPAL_LENGTH].quantile(0.25), 2))
    print("Sepal Width:", round(iris[SEPAL_WIDTH].quantile(
        0.75) - iris[SEPAL_WIDTH].quantile(0.25), 2))
    print("Petal Length:", round(iris[PETAL_LENGTH].quantile(
        0.75) - iris[PETAL_LENGTH].quantile(0.25), 2))
    print("Petal Width:", round(iris[PETAL_WIDTH].quantile(
        0.75) - iris[PETAL_WIDTH].quantile(0.25), 2))

    print("\nStats on All Data")
    print(iris.describe())

    for specie in species:
        print(f"\nStats on {specie.capitalize()}")
        print(iris[iris.species == specie].describe())


def display_one_histogram(ax, iris, column_name, title):
    ax.hist(iris[column_name])
    ax.set_title(title)
    ax.axvline(iris[column_name].median(),
               color="g", label="median", linewidth=5)
    ax.axvline(iris[column_name].mean(),
               color="r", label="mean")
    ax.axvline(iris[column_name].quantile(
        0.25), color="blue", label="Q1", linestyle="dotted")
    ax.axvline(iris[column_name].quantile(
        0.75), color="blue", label="Q3", linestyle="dotted")
    modes = iris[column_name].mode()
    for m in modes:
        ax.axvline(m, color="y", label="mode", linestyle="--")
    ax.legend()


def display_count_plot(data, title):
    sns.countplot(data)
    plt.title(title)
    plt.show()


def display_one_boxplot(ax, iris, column_name, title):
    ax.boxplot(iris[column_name])
    ax.set_title(title)


def display_one_kdeplot(ax, iris, column_name, title):
    sns.kdeplot(data=iris, x=column_name, ax=ax)
    ax.set_title(title)


def display_all_graphs(
        iris, species_names, titles, column_names,
        graph_function, show_legend):

    fig, ax = plt.subplots(1+len(species_names), len(titles), figsize=(14, 10))

    # Show histograms of each column
    for column, (title, column_name) in enumerate(zip(titles, column_names)):
        graph_function(ax[0][column], iris, column_name, title)

    for row, species_name in enumerate(species_names, start=1):
        # Show histograms of each column for each species_name (e.g. "setosa")
        for column, (title, column_name) in enumerate(zip(titles, column_names)):
            data = iris[iris["species"] == species_name]
            graph_function(
                ax[row][column], data, column_name, f"{species_name.capitalize()}-{title}")
            if show_legend:
                ax[row][column].legend()

    fig.tight_layout()
    plt.show()


def display_scatter_combinations(
        iris, titles, column_names, main_title):
    combinations = list(itertools.combinations(zip(titles, column_names), 2))
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))

    for index, combination in enumerate(combinations):
        row = index // 3
        column = index % 3
        ax[row][column].set_title(
            f"{combination[0][0]} vs {combination[1][0]}")
        # ax[row][column].scatter(iris[combination[1][1]],
        #                        iris[combination[0][1]])
        sns.scatterplot(x=iris[combination[1][1]],
                        y=iris[combination[0][1]], hue="species",
                        ax=ax[row][column], data=iris)

        ax[row][column].axvline(iris[combination[1][1]].quantile(
            0.25), color="r", label=f"Q1 {combination[1][0]}")
        ax[row][column].axvline(iris[combination[1][1]].quantile(
            0.75), color="r", label=f"Q3 {combination[1][0]}")

        ax[row][column].axhline(iris[combination[0][1]].quantile(
            0.25), color="g", label=f"Q1 {combination[0][0]}")
        ax[row][column].axhline(iris[combination[0][1]].quantile(
            0.75), color="g", label=f"Q3 {combination[0][0]}")

        ax[row][column].legend()

    fig.suptitle(main_title)
    plt.tight_layout()
    plt.show()


def display_iris_scatter_plots(iris, species_names, titles, column_names):
    # Show scatter plots between every dimension, no filtering
    display_scatter_combinations(iris, titles,
                                 column_names, "All Data")
    for species_name in species_names:
        data = iris[iris["species"] == species_name]
        display_scatter_combinations(data, titles,
                                     column_names, 
                                     f"{species_name.capitalize()} Data")


iris = sns.load_dataset('iris')

species_names = ["setosa", "virginica", "versicolor"]
titles = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
column_names = [SEPAL_LENGTH, SEPAL_WIDTH, PETAL_LENGTH, PETAL_WIDTH]

display_count_plot(iris["species"], "Counts by Species")

display_all_graphs(
    iris, species_names, titles, column_names, display_one_kdeplot, False)

display_all_graphs(
    iris, species_names, titles, column_names, display_one_boxplot, False)

display_all_graphs(
    iris, species_names, titles, column_names, display_one_histogram, True)

display_iris_scatter_plots(iris, species_names, titles, column_names)

print_descriptive_stats(iris, species_names)
