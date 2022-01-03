'''
https://app.pluralsight.com/course-player?clipId=08dbceed-156b-4816-9b8e-7e7d27eaa95a
'''

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras import backend as K
import os
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.pooling import MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = "./machine_learning/supervised/classification/image_processing/"\
    "mnist_fashion/saved_models/keras_clothes_cnn.model"

# Suppress Tensorflow warnings and error messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# How many types of clothes are we predicting
num_classes = 10

batch_size = 128
epochs = 10000

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# different backends (TensorFlow, Theano) use different data formats
# This will avoid "dimension mismatch errors" when switching backends
# Since images are grayscale, the number of channels is 1
# Note that x_test.reshape(2,2) is equivalent to x_test.reshape((2,2))
# unlike np.reshape(x_test, (2,2))
if K.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Original data contains uint32.
# We need to converty and scale
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

# One hot encoding
# Similar to panda's get_dummies()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

# Use 32 3x3 filters;  instead of specifying
# These filters will likely find simple patterns like lines and curves
# a separate relu later (a different way to do the same thing),
# we ask the Conv2D layer to do this for us
# Often used # of filters : 16, 32, 64, 96
# Often used kernel sizes : 3x3, 5x5, 7x7
# We might also want to try padding="same" (instead of default "valid")
# so that "image" doesn't shrink after this layer
# Original demo started with 32, then changed to 64
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation="relu",
                 input_shape=input_shape))

# Using a 2x2 grid, find the "max" while sliding acroos image,
# essentially "shrinking" the image.  In this way, we reduce
# trainable parameters (reduce dimensionality), and also
# help with "translational invariance".  That is, we create
# a more generalized version of the image.  e.g. if a "7"
# was drawn skewed, the max pooling layer would convert
# it to be a more canonical "7".
# If we were using time data, we would use 1D, and if
# we were using video data, we would use 3D
# Width output = ((Width input - Width of pool layer)/Stride) + 1
# In demo, he later removed this layer because
# a paper showed that removing early pooling layers helped performance
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add a couple more CNN layers
# to create filters that detect more complex features built
# from previous simple features/filters

model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Now that we've found image features, let's use
# normal neural network stuff to categorize the pictures
# First, convert 2D images into 1D feature vector
model.add(Flatten())

model.add(Dense(128, activation="relu"))
# Prevent overfitting
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

# "compile" really means set loss, optimizer, and metrics
model.compile(loss=keras.losses.categorical_crossentropy,
              # Why Adadelta and not Adam??
              optimizer=keras.optimizers.Adadelta(),
              metrics=["accuracy"])

my_callbacks = [EarlyStopping(monitor="accuracy", patience=5, mode=max)]

hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 callbacks=my_callbacks,
                 verbose=1,
                 validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
# This will tell us what gets returned by model.evaluate above
# The items are returned in a list
print("Metric Names: ", model.metrics_names)
print("Test Loss: ", score[0])
print("Test Accuracy: ", score[1])

# values for epoch history ; 1,2, ... #epochs
epoch_list = list(range(1, len(hist.history["accuracy"])+1))
plt.plot(epoch_list, hist.history["accuracy"],
         epoch_list, hist.history["val_accuracy"])
plt.legend("Training Accuracy", "Validation Accuracy")
plt.show()
