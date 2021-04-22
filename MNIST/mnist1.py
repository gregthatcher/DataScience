'''
Experimenting with Mnist data set
Ideas from:
https://towardsdatascience.com/mnist-cnn-python-c61a5bce7a19
'''
from keras.datasets import mnist
import keras
from keras.utils import to_categorical

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
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)