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

MODEL_PATH = "./models/minst1.model"

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
print('Test loss', test_loss)
print('Test accuracy', test_acc)

predictions = model.predict(test_X)
final_prediction = np.argmax(np.round(predictions[0]))
print(final_prediction)


plt.imshow(test_X[0], cmap=plt.get_cmap('gray'))
plt.title(f"Model Predicted {final_prediction}")

plt.show()