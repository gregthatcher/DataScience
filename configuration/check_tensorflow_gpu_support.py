'''
Idea from https://www.youtube.com/watch?v=IubEtS2JAiY&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=2
For more info on setting up NVIDIA Cuda drivers, see
https://deeplizard.com/learn/video/IubEtS2JAiYpy
'''

import tensorflow as tf
print("Number of Available GPUs: ", len(tf.config.experimental.list_physical_devices("GPU")))