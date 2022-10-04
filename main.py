import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

from keras import backend as K

#Image properties
height, width, depth = 320, 240, 3
outputs = 5


#Template for image
image = np.zeros((height,width,depth))
normalized_image = np.divide(image, 255)
flattened_image = np.reshape(normalized_image, [-1,np.product(normalized_image.shape)])

print(image.shape)
print(normalized_image.shape)
print(flattened_image.shape)

def make_model(image_resolution, informational_neurons, outputs, dropout_rate)
    image_data = keras.Input(shape=(image_resolution), name="image_data")
    information = keras.Input(shape=(informational_neurons), name="information")

    all_inputs = [image_data, information]

    #input image_data into cnn
    x = layers.Conv2D(32, 3, padding="same")(image_data)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Dropout(dropout_rate)(x)

    for size in [64, 128, 256]:
        x = layers.Conv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        x = layers.Dropout(dropout_rate)(x)

    x = layers.Flatten()(x)

    #input information and flattened image_data layer into dense layer
    concatenated_layers = layers.Concatenate([x, information])

    x = layers.Dense(32, activation="relu")(cancatenated_layers)
    x = layers.Dropout(dropout_rate)(x)

    #get output
    output = layers.Dense(outputs, activation="relu")(x)

    model = keras.Model(all_inputs, output)
    return model

model = make_model([height, width,depth], outputs, outputs, 0.2)

#compile model (update this please)
model.compile(
    optimizer = keras.optimizer.Adam(1e-3),
    loss = "categorical_crossentropy",    
    metrics = ["accuracy"],
)


