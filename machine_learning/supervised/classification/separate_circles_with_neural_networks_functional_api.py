'''
This is similar to separate_circles_with_neural_networks.py,
but it uses the keras "functional api" which is more
flexible than the "Sequential" model, allowing for more
complex neural network architectures, including
multiple inputs (at different layers), "iterative layers",
and "multiple outputs".
See : https://app.pluralsight.com/course-player?clipId=dc7c646b-d092-4a28-bfdb-01b97e477e6d
'''


import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.callbacks import EarlyStopping

# Get rid of tensorflow info and warnings
# Remove this if you want to see if the GPU
# is used or not, or if you want optimization
# recommendations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def plot_data(X, y):
    plt.plot(X[y == 0, 0], X[y == 0, 1], "ob", alpha=0.5)
    plt.plot(X[y == 1, 0], X[y == 1, 1], "xr", alpha=0.5)
    plt.legend(["0", "1"])


# Here, we do a trick with a countour
# the model will give us predictions between 0 and 1
# Far from the boundary, the values will be 0 or 1
# Near the boundary, the values will have other values
# We use these model values as the "height"' of the contour
# so we get different colors at the boundary
def plot_decision_boundary(model, X, y):
    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1

    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)

    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]

    # Make predictions with model, and then reshape output
    # so that contourf can plot it
    c = model.predict(ab)
    Z = c.reshape(aa.shape)

    plt.figure(figsize=(12, 8))
    plt.contourf(aa, bb, Z, cmap="bwr", alpha=0.2)

    plot_data(X, y)

    return plt


X, y = make_circles(n_samples=1000, factor=.6, noise=0.1, random_state=42)

plot_data(X, y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Old way
# model = Sequential()
# model.add(Dense(4, activation="tanh", input_shape=(2,)))
# model.add(Dense(4, activation="tanh"))
# model.add(Dense(1, activation="sigmoid"))

# New, Functional API way
inputs = Input(shape=(2,))
# Hidden layers
x = Dense(4, activation="tanh", name="Hidden-1")(inputs)
x = Dense(4, activation="tanh", name="Hidden-2")(x)
# Output layer
o = Dense(1, activation="sigmoid", name="Output-Layer")(x)
model = Model(inputs=inputs, outputs=o)
# Do I need this print??
print(model.summary())
# 3.) Compile Model
# Minimize cross entropy for a binary.
# Maximize for accuracy.
# Use Adam optimizer to minimize loss
# binary_crossentropy is used to calculate loss
# Accuracy is what we want to optimize
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])

# Let's add "early stopping"
# Note that "val_accuracy" probably makes more sense in the real world
my_callbacks = [EarlyStopping(monitor="accuracy", patience=5, mode=max)]
# 4.) Train Model (via fit)
# Verbose = 1 for progress bar
# BE SURE TO SHUFFLE DATA BEFORE USING validation_split
# see https://www.youtube.com/watch?v=U8Ixc2OLSkQ&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=7
model.fit(X_train, y_train, epochs=100000, verbose=1, callbacks=my_callbacks)


# 5.) Evaluate Performance
# Get loss and accuracy on test data
eval_result = model.evaluate(X_test, y_test)
info = f"\n\nTest loss:{eval_result[0]:.6f}\nTest accuracy: {eval_result[1]}"
print(info)

plot_decision_boundary(model, X, y)
plt.text(0, 6, info)
plt.show()

y_infer = model.predict(X)
y_infer = np.round(y_infer)
# print(y_infer)
# See https://towardsdatascience.com/understanding-the-confusion-matrix-from-scikit-learn-c51d88929c79
print("TN", "FP")
print("FN", "TP")
cm = confusion_matrix(y_true=y, y_pred=y_infer)
print(cm)
