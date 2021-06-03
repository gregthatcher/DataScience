# Ideas from:
# https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/

'''
One of my machines has an AMD GPU, so
I can't use TensorFlow, so I can't use keras, 
so I can't easily download MNIST using keras.datasets,
so I have to do all this nonsense to get the dataset
Ideas from: https://stackoverflow.com/questions/40690203/how-can-i-import-the-mnist-dataset-that-has-been-manually-downloaded
'''

import gzip
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

print(os.getcwd())
f = gzip.open('./MNIST/data/mnist.pkl.gz', 'rb')
data = pickle.load(f, encoding='bytes')
f.close()
(x_train, y_train), (x_test, y_test) = data
# print(x_train)
# print(data)
# there are 60,000 examples in the training dataset,
# and 10,000 in the test dataset,
# and images are square with 28×28 pixels
print("Type of Data", type(data))
print("Data Shape", np.ndim(data))
print("x_train", x_train.shape)
print("y_train", y_train.shape)
print("x_test", x_test.shape)
print("y_test", y_test.shape)


greg = x_train.reshape((x_train.shape[0], 28, 28, 1))
print("Greg", greg.shape)
print("x", x_train.shape)


fig, axs = plt.subplots(nrows=3, ncols=3)
# Let's show some images so we know what we're looking at
i = 0
for row in axs:
    for ax in row:
        ax.imshow(x_train[i], cmap=plt.get_cmap('gray'))
        i += 1

# display the plots
plt.show()
