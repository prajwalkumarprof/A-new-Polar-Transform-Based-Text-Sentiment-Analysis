

import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float
 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import io 
from PIL import Image 


from PolarBase import loaddataset




def create_cnn_model_1(input_shape):
    inputs = Input(shape=input_shape)
    x = layers.Conv1D(256, 2, activation='relu')(inputs)
    #x = layers.Conv1D(128, 2, activation='relu')(x)
    x = layers.Dropout(0)(x)
    x = layers.Conv1D(128, 1, activation='relu')(x)
    x = layers.Dropout(0)(x)
     
   # x = layers.MaxPooling1D()(x)
    x = layers.BatchNormalization()(x)#
    x = layers.Dense(64 ,activation='relu')(x)
    x = layers.Dense(256 ,activation='relu')(x)
   
    return Model(inputs=inputs, outputs=x)

input_shape = (2,256) # histo polar
model = create_cnn_model_1(input_shape)


HistofPolorTR = loaddataset()
print(HistofPolorTR)
HistofPolorTR= np.array(HistofPolorTR)  
x_train, x_test= train_test_split(HistofPolorTR, test_size=0.2) #histo polar

model.compile(
  optimizer='SGD',
  loss='mean_absolute_error',
  metrics=['accuracy'])

history =model.fit(
  x_train,
  x_train,
  #validation_data=x_test.all(),
  epochs=9
)


acc = history.history['accuracy']
print(f"Train accuracy: {acc}")
#model.save("polarTnRmodel.h5")
model.save("polarTnRmodel.keras")