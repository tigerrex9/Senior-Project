import os
import pandas as pd
import numpy as np

#import picamera

from tensorflow import keras

model = keras.models.load_model("saved_model")


def normalize_image(image):
    return (np.divide(image, 255))

def flatten_image(image):
    return (np.reshape(image, [-1, np.product(image.shape)]))

image = np.zeros((320, 240, 3))
normalized_image = normalize_image(image)
flattened_image = flatten_image(normalized_image)

