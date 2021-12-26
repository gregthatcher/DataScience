import tensorflow as tf
print("Number of Available GPUs: ", len(tf.config.experimental.list_physical_devices("GPU")))