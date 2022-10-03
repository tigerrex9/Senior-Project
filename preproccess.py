import os
import pandas as pd
import numpy as np

image = np.zeros((320, 240, 3))
normalized_image = np.divide(image, 255)
flattened_image = np.reshape(normalized_image, [-1,np.product(normalized_image.shape)])

print(image.shape)
print(normalized_image.shape)
print(flattened_image.shape)

