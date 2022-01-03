'''
Use transfer learning to classify images of dogs and cata
Ideas from Kurata lecture https://app.pluralsight.com/course-player?clipId=eee015a8-88fb-4c10-99ec-5663dc5579a5
His ideas came from https://app.pluralsight.com/course-player?clipId=eee015a8-88fb-4c10-99ec-5663dc5579a5
Use pre-trained "Inception model" to classify dogs and cats
Data from https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
'''

import glob
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling1D
from keras.preprocessing.image import ImageDataGenerator
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_num_files(path):
    if not os.path.exists(path):
        return 0

    return sum([len(files) for r, d, files in os.walk(path)])


def get_num_folders(path):
    if not os.path.exists(path):
        return 0

    return sum([len(d) for r, d, files in os.walk(path)])

