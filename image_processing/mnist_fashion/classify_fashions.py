'''
Classification of MNIST Fashions using keras

Lots of ideas from:
https://medium.com/nerd-for-tech/how-to-train-neural-networks-for-image-classification-part-1-21327fe1cc1
'''
# Note that keras is now _part_ of TensorFlow
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_PATH = "./image_processing/mnist_fashion/models/minst1.model"

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

try:
    model = keras.models.load_model(MODEL_PATH)
except OSError:
    # Final layer has 10 nodes as we have 10 classes
    model = keras.models.Sequential([keras.layers.Flatten(input_shape=[28, 28]),
                                    keras.layers.Dense(300, activation="relu"),
                                    keras.layers.Dense(100, activation="relu"),
                                    keras.layers.Dense(100, activation="relu"),
                                    keras.layers.Dense(100, activation="relu"),
                                    keras.layers.Dense(10, activation="softmax")])

    print("Model Summary:")
    print(model.summary())

    # "sparse_categorical_crossentropy" because we have sparse (distinct) labels

    model.compile(loss="sparse_categorical_crossentropy",
                optimizer="sgd",
                metrics=["accuracy"])

    # The accuracy shown at each epoch is the percentage correct
    # The loss is the cross-entropy loss
    history = model.fit(X_train,
                        y_train,
                        epochs=10,
                        validation_data=(X_validation, y_validation))

    model.save(MODEL_PATH)

    pd.DataFrame(history.history).plot(figsize = (16, 10))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
    
# How good is our model?
model.evalute(X_test, y_test)
