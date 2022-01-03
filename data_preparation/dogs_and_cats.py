'''
Data for "transfer learning demo" from https://app.pluralsight.com/course-player?clipId=eee015a8-88fb-4c10-99ec-5663dc5579a5

We have copied train.zip data from https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data?select=train.zip
into our downloads directory

Per instructions, we want to copy 1000 cat images and 1000 dog images to the "train" folder,
and 400 cat images and 400 dog images to the "validate" folder
'''

import glob
import os
import random
from shutil import copy

DATA_DIRECTORY = "C:\\Users\\greg\\Downloads\\dogs-vs-cats-redux-kernels-edition\\train\\train"
TARGET_DIRECTORY = os.path.abspath(".\\data\\raw_data\\dogs_and_cats")
TRAIN_DIRECTORY = TARGET_DIRECTORY + "\\train\\"
VALIDATE_DIRECTORY = TARGET_DIRECTORY + "\\validate\\"


def remove_old_files(directory):
    files = glob.glob(directory + "*")
    for f in files:
        os.remove(f)

def copy_sample_of_species_files(input_glob):
    all_species_files = glob.glob(DATA_DIRECTORY + input_glob)
    # Let's get 1000 for train, 400 for test, equals 1400 total
    sample_of_species_files = random.sample(all_species_files, 1400)

    counter = 0
    for file in sample_of_species_files:
        counter += 1
        if counter <= 1000:
            copy(file, TRAIN_DIRECTORY)
        else:
            copy(file, VALIDATE_DIRECTORY)


print("Data Directory: ", DATA_DIRECTORY)
print("Target Directory: ", TARGET_DIRECTORY)
print("Train Directory: ", TRAIN_DIRECTORY)
print("Current Working Directory: ", os.getcwd())

remove_old_files(TRAIN_DIRECTORY)
remove_old_files(VALIDATE_DIRECTORY)

copy_sample_of_species_files("\\dog*")
copy_sample_of_species_files("\\cat*")
