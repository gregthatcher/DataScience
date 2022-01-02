'''
Classification of MNIST Fashions using keras

Lots of ideas from:
https://medium.com/nerd-for-tech/how-to-train-neural-networks-for-image-classification-part-1-21327fe1cc1
'''
# Note that keras is now _part_ of TensorFlow
# We have the option to just install TensorFlow, or install Keras
# separately, and use the "api" to access TensorFlow or ??
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import pandas as pd
from tensorflow.keras import callbacks
from keras.utils import plot_model

MODEL_PATH = "./machine_learning/supervised/classification/image_processing/"\
    "mnist_fashion/saved_models/keras_clothes.model"

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print("Type:", type(X_train_full))

# Let's save 20% of our test set for validation
n_validation_rows = int(X_train_full.shape[0] * .2)
X_validation, X_train = X_train_full[:
                                     n_validation_rows], X_train_full[n_validation_rows:]
y_validation, y_train = y_train_full[:
                                     n_validation_rows], y_train_full[n_validation_rows:]
print("Number of Training Rows:", X_train.shape[0])
print("Number of Validation Rows:", X_validation.shape[0])
print("Number of Test Rows:", X_test.shape[0])

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress",
               "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Show a sample of some data
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train_full[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train_full[i]])
plt.show()

print("Old Min:", X_train_full.min())
print("Old Max:", X_train_full.max())

# Now, let's normalize (not standardize)
X_validation = X_validation / 255.0
X_train = X_train / 255.0
X_test = X_test / 255.0
print("New Min:", X_validation.min())
print("New Max:", X_validation.max())

# Alternatively, we could use .save_weights() and .load_weights()
# and then train the model every time (it won't take long after the first time)
try:
    model = keras.models.load_model(MODEL_PATH)
except OSError:
    # Final layer has 10 nodes as we have 10 classes
    model = keras.models.Sequential([keras.layers.Flatten(input_shape=[28, 28]),
                                    keras.layers.Dense(300, activation="relu"),
                                    keras.layers.Dense(100, activation="relu"),
                                    keras.layers.Dense(100, activation="relu"),
                                    keras.layers.Dense(100, activation="relu"),
                                    keras.layers.Dense(10, activation="softmax"
                                                       )])

    print("Model Summary:")
    print(model.summary())

    # "sparse_categorical_crossentropy" because we have sparse (distinct)
    # labels
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])

    my_callbacks = [keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, mode=max)]
    # The accuracy shown at each epoch is the percentage correct
    # The loss is the cross-entropy loss
    # BE SURE TO SHUFFLE DATA BEFORE USING validation_split
    # see https://www.youtube.com/watch?v=U8Ixc2OLSkQ&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=7
    history = model.fit(X_train,
                        y_train,
                        epochs=10000,
                        callbacks=my_callbacks,
                        verbose=2,
                        validation_data=(X_validation, y_validation))

    model.save(MODEL_PATH)

    pd.DataFrame(history.history).plot(figsize=(16, 10))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

# How good is our model?
print("Evaluate:")
model.evaluate(X_test, y_test)

filename = "mnist_model.png"
model.summary()
# TODO: Can't get this to work??
# plot_model(model, to_file=filename,
#           show_shapes=True, show_layer_names=True)
#im = img.imread(filename)
# plt.imshow(im)
