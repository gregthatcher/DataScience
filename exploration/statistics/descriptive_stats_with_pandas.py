'''
This module contains routines for doing descriptive statistics (not inferential) using pandas dataframes
'''
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

SEPAL_LENGTH = "sepal_length"
SEPAL_WIDTH = "sepal_width"
PETAL_LENGTH = "petal_length"
PETAL_WIDTH = "petal_width"
SPECIES = "species"

Z_VALUE_FOR_95_PERCENT_CONFIDENCE = 1.96


def print_column_stats_by_species(iris, column):
    print("\nGrouped by Species Stats")
    print(column.capitalize().replace("_", " "))
    grouped_data = iris.groupby([SPECIES])[column].agg([
        'count', 'mean', 'std'])
    confidence_interval_95_high = []
    confidence_interval_95_low = []
    for i in grouped_data.index:
        count, mean, std = grouped_data.loc[i]
        # 1.96 is Z value for 95% confidence
        confidence_interval_95_high.append(
            mean + Z_VALUE_FOR_95_PERCENT_CONFIDENCE * (std/math.sqrt(count)))
        confidence_interval_95_low.append(
            mean - Z_VALUE_FOR_95_PERCENT_CONFIDENCE * (std/math.sqrt(count)))

    grouped_data["ci95_Low"] = confidence_interval_95_low
    grouped_data["ci95_High"] = confidence_interval_95_high

    print("(Remember that Confidence is higher with more data and less std)")
    print(grouped_data)


def print_descriptive_stats(iris, species):
    print("Type: ", type(iris))

    print("Pandas Version: ", pd.__version__)

    print("\nShape of DataFrame:", iris.shape)
    print("\nData Types:")
    print(iris.dtypes)
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

    print("\nGroup By Species")
    species_groupby = iris.groupby("species", as_index=False)
    print(species_groupby.head())
    print("\nGroup By Summary Stats (transposed)")
    print(species_groupby.describe().T)

    print("\nSkewness:")
    print("(Measure of Asymetry; Normal Distribution has skew of 0)")
    print(iris.skew())

    print("\nKurtosis:")
    print("(How likely are extreme events?  Normal Distribution has kurtosis of 3)")
    print(iris.kurtosis())

    print("\nCovariances:")
    print(iris.cov())

    print("\nCorrelations:")
    print(iris.corr())

    print_column_stats_by_species(iris, PETAL_WIDTH)
    print_column_stats_by_species(iris, PETAL_LENGTH)
    print_column_stats_by_species(iris, SEPAL_WIDTH)
    print_column_stats_by_species(iris, SEPAL_LENGTH)


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


def display_bivariate_boxplots(iris, species_names, titles, column_names):
    fig, ax = plt.subplots(len(column_names), 1, figsize=(14, 10))
    for column, (title, column_name) in enumerate(zip(titles, column_names)):
        sns.boxplot(x=SPECIES, y=column_name, data=iris, ax=ax[column])
        ax[column].set_title(title)

    fig.tight_layout()
    plt.show()


def display_bivariate_violin_and_swarm_plots(
        iris, species_names, titles, column_names):
    fig, ax = plt.subplots(len(column_names), 1, figsize=(14, 10))
    for column, (title, column_name) in enumerate(zip(titles, column_names)):
        sns.violinplot(x=SPECIES, y=column_name, data=iris, ax=ax[column])
        sns.swarmplot(x=SPECIES, y=column_name,
                      data=iris, ax=ax[column], color="w")
        ax[column].set_title(title)

    fig.tight_layout()
    plt.show()


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
            data = iris[iris[SPECIES] == species_name]
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
                        y=iris[combination[0][1]], hue=SPECIES,
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


def display_correlations(data):
    plt.figure(figsize=(12, 8))
    corr = data.corr()
    # TODO: What does this code mean?
    # Only show "triangle"
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, cmap="YlOrBr", mask=mask)
    plt.title("Correlations")
    plt.show()


def display_iris_scatter_plots(iris, species_names, titles, column_names):
    # Show scatter plots between every dimension, no filtering
    display_scatter_combinations(iris, titles,
                                 column_names, "All Data")
    for species_name in species_names:
        data = iris[iris[SPECIES] == species_name]
        display_scatter_combinations(data, titles,
                                     column_names,
                                     f"{species_name.capitalize()} Data")


iris = sns.load_dataset('iris')

species_names = ["setosa", "virginica", "versicolor"]
titles = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
column_names = [SEPAL_LENGTH, SEPAL_WIDTH, PETAL_LENGTH, PETAL_WIDTH]

print_descriptive_stats(iris, species_names)

raise Exception("Stop")
# TODO: Research this, and fix
# This should show two "kde" plots, colored by species, on the same plot
# sns.FacetGrid(iris, hue="species", height=5)\
#    .map(sns.displot, 'sepal_width')\
#    .add_legend()
# plt.show()

display_count_plot(iris[SPECIES], "Counts by Species")

display_all_graphs(
    iris, species_names, titles, column_names, display_one_boxplot, False)

display_bivariate_boxplots(iris, species_names, titles, column_names)
display_bivariate_violin_and_swarm_plots(
    iris, species_names, titles, column_names)

# KDE is better for showing the shape (compared to Histograms)
display_all_graphs(
    iris, species_names, titles, column_names, display_one_kdeplot, False)

# But, Histograms are better for viewing outliers
display_all_graphs(
    iris, species_names, titles, column_names, display_one_histogram, True)

display_correlations(iris)

display_iris_scatter_plots(iris, species_names, titles, column_names)
