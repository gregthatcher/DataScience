'''
Use transfer learning to classify images of dogs and cata
Ideas from Kurata lecture https://app.pluralsight.com/course-player?clipId=eee015a8-88fb-4c10-99ec-5663dc5579a5
His ideas came from https://app.pluralsight.com/course-player?clipId=eee015a8-88fb-4c10-99ec-5663dc5579a5
Use pre-trained "Inception model" to classify dogs and cats
Data from https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
'''

import matplotlib.pyplot as plt
# InceptionV3 is the pre-trained model we will "transfer" from
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
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


# Define image generators that will rotate slightly, shift up, down, left,
# right, shear, zoom in, or flip horitontally
# We can use this to augment our training data, helping with
# "translational invariance"
def create_img_generator():
    return ImageDataGenerator(
        # Note that we imported "preprocess_input" from InceptionV3 above
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )


IMAGE_WIDTH, IMAGE_HEIGHT = 299, 299
TRAINING_EPOCHS = 2
BATCH_SIZE = 32
# Number of neurons in Fully Connected Classification Layer
NUMBER_FC_NEURONS = 1024

train_dir = "./data/raw_data/dogs_and_cats/train"
validate_dir = "./data/raw_data/dogs_and_cats/validate"

num_train_samples = get_num_files(train_dir)
num_classes = get_num_folders(train_dir)
nun_validate_samples = get_num_files(validate_dir)

print("Number of Samples : ", num_train_samples)
print("Number of Classes : ", num_classes)
print("Num Validation Samples : ", nun_validate_samples)

num_epochs = TRAINING_EPOCHS
batch_size = BATCH_SIZE

train_image_gen = create_img_generator()
test_image_gen = create_img_generator()

# Connect image generator to our _train_ source images
# The image generator will alter these images
train_generator = train_image_gen.flow_from_directory(
    train_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    seed=42  # set seed for reproducibility
)

# Connect image generator to our _validation_ source images
# The image generator will alter these images
validation_generator = train_image_gen.flow_from_directory(
    validate_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    seed=42  # set seed for reproducibility
)

# include_top=False means exclude final Fully Connected (FC) layer
# Get model from "imagenet" competition
inceptionV3_base_model = InceptionV3(weights="imagenet", include_top=False)
# FC = "Fully Connected"
print("Inception v3 basd model without last Fully Connected layer (FC) loaded")

x = inceptionV3_base_model.output
x = GlobalAveragePooling2D()(x)
# New FC layer, random init
x = Dense(NUMBER_FC_NEURONS, activation="relu")(x)
# New softmax layer
predictions = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=inceptionV3_base_model.input, outputs=predictions)

print(model.summary())
