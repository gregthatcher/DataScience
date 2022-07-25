# Ideas from https://campus.datacamp.com/courses/statistical-thinking-in-python-part-1/graphical-exploratory-data-analysis?ex=15

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def show_histogram(ax, species, column_name, x_label, bins):
    ax.hist(iris[iris.species == species][column_name], bins=bins)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    ax.set_title(species)


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y


iris = sns.load_dataset('iris')

print(iris.head())
# Rule of thumb, number of bins should be square root
# of length of data
n_bins = int(np.sqrt(iris.shape[0]))


fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
show_histogram(ax[0][0], "setosa", "sepal_length", "Sepal Length", n_bins)
show_histogram(ax[0][1], "versicolor", "sepal_length", "Sepal Length", n_bins)
show_histogram(ax[0][2], "virginica", "sepal_length", "Sepal Length", n_bins)

show_histogram(ax[1][0], "setosa", "sepal_width", "Sepal Length", n_bins)
show_histogram(ax[1][1], "versicolor", "sepal_width", "Sepal Length", n_bins)
show_histogram(ax[1][2], "virginica", "sepal_width", "Sepal Length", n_bins)

plt.show()

fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
show_histogram(ax[0][0], "setosa", "petal_length", "Sepal Length", n_bins)
show_histogram(ax[0][1], "versicolor", "petal_length", "Sepal Length", n_bins)
show_histogram(ax[0][2], "virginica", "petal_length", "Sepal Length", n_bins)

show_histogram(ax[1][0], "setosa", "petal_width", "Petal Length", n_bins)
show_histogram(ax[1][1], "versicolor", "petal_width", "Petal Length", n_bins)
show_histogram(ax[1][2], "virginica", "petal_width", "Petal Length", n_bins)

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(12, 8))

sns.swarmplot(x="species", y="sepal_width", data=iris, ax=ax[0][0])
sns.swarmplot(x="species", y="sepal_length", data=iris, ax=ax[0][1])
sns.swarmplot(x="species", y="petal_width", data=iris, ax=ax[1][0])
sns.swarmplot(x="species", y="petal_length", data=iris, ax=ax[1][1])
plt.show()

fig, ax = plt.subplots(1, 4, figsize=(12, 8))

def plot_ecdf_graph(ecdf, iris, column, ax):
    # Compute ECDF for versicolor data: x_vers, y_vers
    x_vers, y_vers = ecdf(iris[iris.species == "versicolor"][column])
    x_set, y_set = ecdf(iris[iris.species == "setosa"][column])
    x_virg, y_virg = ecdf(iris[iris.species == "virginica"][column])

# Generate plot
    ax.plot(x_set, y_set, marker=".", linestyle="none")
    ax.plot(x_vers, y_vers, marker=".", linestyle="none")
    ax.plot(x_virg, y_virg, marker=".", linestyle="none")

    ax.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
    ax.set_xlabel(f'{column} (cm)')
    ax.set_ylabel('ECDF')

    plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')


plot_ecdf_graph(ecdf, iris, "petal_length", ax[0])
plot_ecdf_graph(ecdf, iris, "petal_width", ax[1])
plot_ecdf_graph(ecdf, iris, "sepal_width", ax[2])
plot_ecdf_graph(ecdf, iris, "sepal_length", ax[3])


# Display the plot
plt.show()
