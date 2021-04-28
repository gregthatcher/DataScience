'''
Experimenting with Mnist data set
Ideas from:
https://towardsdatascience.com/mnist-cnn-python-c61a5bce7a19
'''
from keras.datasets import mnist
import keras
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = "./mnist/models/minst1.model"

(train_X, train_y), (test_X, test_y) = mnist.load_data()
#from mnist import MNIST

print("Initialial Shape ", train_X.shape)
train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))
print("Final Shape ", train_X.shape)

# Modifying the values of each pixel such that they range from 0 to 1 will 
# improve the rate at which our model learns
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255
test_X = test_X / 255

# Use one-hot encoding (if we were using DataFrames, we could use "dummies")
train_Y_one_hot = to_categorical(train_y)
test_Y_one_hot = to_categorical(test_y)
print(train_Y_one_hot.shape)
print(test_Y_one_hot.shape)


try:
    model = keras.models.load_model(MODEL_PATH)
except OSError:
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

    model.fit(train_X, train_Y_one_hot, batch_size=64, epochs=2)

    model.save(MODEL_PATH)

test_loss, test_acc = model.evaluate(test_X, test_Y_one_hot)

num_rows = 2
num_columns = 3
fig, ax = plt.subplots(num_rows, num_columns, figsize=(10, 6.5))

predictions = model.predict(test_X)

fig.suptitle(f"Test Loss {round(test_loss, 2)}; Test Accuracy {round(test_acc, 2)}")
counter = 0
for row in range(num_rows):
    for column in range(num_columns):
        final_prediction = np.argmax(np.round(predictions[counter]))

        ax[row][column].imshow(test_X[counter], cmap=plt.get_cmap('gray'))
        ax[row][column].set_title(f"Model Predicted {final_prediction}")

        counter += 1

plt.show()