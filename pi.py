import os
import pandas as pd
import numpy as np

import picamera

from tensorflow import keras

model = keras.models.load_model("saved_model")

def normalize_image(image):
    return (np.divide(image, 255))

def flatten_image(image):
    return (np.reshape(image, [-1, np.product(image.shape)]))

with picamera.PiCamera() as camera:
    camera.resolution = (320, 240)
    camera.framerate = 24
    time.sleep(2)
    image = np.empty((240, 320, 3), dtype=np.uint8)
    camera.capture(image, 'rgb')


prediction = model.predict(image)

print(prediction.shape)
print(prediction)
