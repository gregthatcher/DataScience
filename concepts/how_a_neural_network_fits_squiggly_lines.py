"""
Ideas from https://www.youtube.com/watch?v=CqOfi41LfDw
Josh Stammer explains how neural networks are actually giant squiggly
fitting machines.  In the code below, I try to show a a simple
neural network builds a "squiggly" curve out of portions of
the activation function.
"""

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np


DOSAGE_START_X = 0.0
DOSAGE_END_X = 1.0

WEIGHT_1 = -34.4
BIAS_1 = 2.14
WEIGHT_2 = -2.52
BIAS_2 = 1.29


# max(0,x)
def relu(x):
    return np.maximum(0, x)


# S(x) = 1 / (1+e^-x)
def sigmoid(x):
    return 1 / (1 + (1 + np.exp(-x)))


# Convert a raw value into a posterior probability
def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo / expo_sum


# Idea from : https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
# log(1 + exp(x))
def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def draw_activation_part(x, ax, activation_function, weight, bias, title):
    x1 = (DOSAGE_START_X * weight) + bias
    x2 = (DOSAGE_END_X * weight) + bias
    y = activation_function(x)
    ax.plot(x, y)
    title = f"{title} (weight = {weight} bias = {bias})"
    ax.set_title(title)
    height = max(activation_function(x1), activation_function(x2))
    ax.add_patch(
        Rectangle((x2, 0), abs(x2 - x1), height,
                  facecolor="none", edgecolor="red")
    )


def display_portions_of_activation_functions(
    weight_1, bias_1, weight_2, bias_2, activation_function, ax, data_x
):
    first_x = (data_x * weight_1) + bias_1
    second_x = (data_x * weight_2) + bias_2

    ax.plot(data_x, activation_function(first_x),
            label=f"1st Node Weights * {weight_1} + {bias_1}")
    ax.plot(data_x, activation_function(second_x),
            label=f"2nd Node Weights * {weight_2} + {bias_2}")
    ax.set_title("First Layer Activations")
    ax.legend()


def display_final_sums(
    weight_1, bias_1, weight_2, bias_2, activation_function, ax, data_x,
    output_weight_1, output_weight_2, final_bias
):
    first_x = (data_x * weight_1) + bias_1
    second_x = (data_x * weight_2) + bias_2

    first_part = output_weight_1 * activation_function(first_x)
    ax.plot(data_x, first_part,
            label=f"1st Node Activation * {output_weight_1}")
    second_part = output_weight_2 * activation_function(second_x)
    ax.plot(data_x, second_part,
            label=f"2nd Node Activation * {output_weight_2}")
    ax.plot(data_x, first_part + second_part + final_bias,
            label=f"Final Sum + {final_bias}")
    ax.set_title("Final Curve Fits Data Points")
    ax.legend()


def draw_original_points(ax):
    size = 0.1
    color = "yellow"
    alpha = 1
    edge_color = "black"
    line_width = 1
    radius = 0.05
    ax.add_patch(plt.Circle((0, 0), radius=radius, facecolor=color,
                 alpha=alpha, edgecolor=edge_color, linewidth=line_width))
    ax.add_patch(plt.Circle((0.05, 0), radius=radius, facecolor=color,
                 alpha=alpha, edgecolor=edge_color, linewidth=line_width))
    ax.add_patch(plt.Circle((0.1, 0), radius=radius, facecolor=color,
                 alpha=alpha, edgecolor=edge_color, linewidth=line_width))

    ax.add_patch(plt.Circle((0.5, 1.0), radius=radius, facecolor=color,
                 alpha=alpha, edgecolor=edge_color, linewidth=line_width))
    ax.add_patch(plt.Circle((0.55, 1.0), radius=radius, facecolor=color,
                 alpha=alpha, edgecolor=edge_color, linewidth=line_width))

    ax.add_patch(plt.Circle((1, 0), radius=radius, facecolor=color,
                 alpha=alpha, edgecolor=edge_color, linewidth=line_width))
    ax.add_patch(plt.Circle((1.05, 0), radius=radius, facecolor=color,
                 alpha=alpha, edgecolor=edge_color, linewidth=line_width))
    ax.add_patch(plt.Circle((1.1, 0), radius=radius, facecolor=color,
                 alpha=alpha, edgecolor=edge_color, linewidth=line_width))


plt.style.use("seaborn")

fig, ax = plt.subplots(2, 4, figsize=(14, 10))

fig.suptitle("Bulding a Squiggle with Activation Functions")

activation_x = np.linspace(-32, 5, 101)
data_x = np.linspace(0, 1, 101)

draw_activation_part(activation_x, ax[0][0], softplus, WEIGHT_1, BIAS_1,
                     "SoftPlus")
draw_activation_part(activation_x, ax[1][0], relu, WEIGHT_1, BIAS_1, "Relu")
# I don't have weights (from video abvove) for sigmoid or softmax
# draw_activation_part(x, ax[2][0], sigmoid, WEIGHT#_1, BIAS_1, "Sigmoid")
# draw_activation_part(x, ax[3][0], softmax, WEIGHT_1, BIAS_1, "Softmax")

draw_activation_part(activation_x, ax[0][1], softplus, WEIGHT_2, BIAS_2,
                     "SoftPlus")

draw_activation_part(activation_x, ax[1][1], relu, WEIGHT_2, BIAS_2,
                     "Relu")

# I don't have weights (from video abvove) for sigmoid or softmax
# draw_activation_part(x, ax[2][1], sigmoid, WEIGHT_2, BIAS_2, "Sigmoid")
# draw_activation_part(x, ax[3][1], softmax, WEIGHT_2, BIAS_2, "Softmax")

display_portions_of_activation_functions(
    WEIGHT_1, BIAS_1, WEIGHT_2, BIAS_2, softplus, ax[0][2], data_x
)
draw_original_points(ax[0][2])

display_portions_of_activation_functions(
    WEIGHT_1, BIAS_1, WEIGHT_2, BIAS_2, relu, ax[1][2], data_x
)
draw_original_points(ax[1][2])

display_final_sums(WEIGHT_1, BIAS_1, WEIGHT_2, BIAS_2, softplus, ax[0][3],
                   data_x, -1.3, 2.28, -.58)
draw_original_points(ax[0][3])

display_final_sums(WEIGHT_1, BIAS_1, WEIGHT_2, BIAS_2, relu, ax[1][3],
                   data_x, -1.3, 2.28, -.58)
draw_original_points(ax[1][3])

plt.tight_layout()
plt.show()
