

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

#image_folder = '../Datasetww/'
#image_folder = '../DatasetwithR/'
image_folder = '../DatasetRGB/'
Directorylist = []
batch_size = 32
img_height = 240
img_width = 240

radius = 110 # workeed for input polar tested
angle = 35

for pathb in os.listdir(image_folder):
    Directorylist.append(pathb)
print(Directorylist)


imageslist = []
for path in Directorylist:
    for filename in os.listdir(image_folder+""+path):
        if filename.endswith(".png"):
            imageslist.append(image_folder+""+path+"/"+filename)


images = [img_to_array(load_img(img_path, target_size=(240, 240))) for img_path in imageslist]

images = np.array(images)
images = images / 255.0 # required to load images

#fig= plt.subplots(1,  figsize=(8, 8))
 
#plt.imshow(images[1])
#plt.show()
#images[12] ,images[10]

image = img_as_float(images[10])
rotated = rotate(image, angle)
image_polar = warp_polar(image, radius=radius, channel_axis=-1)

rotated_polar = warp_polar(rotated, radius=radius, channel_axis=-1)

rotated2 = rotate(image, 60)
rotated_polar2 = warp_polar(rotated2, radius=radius, channel_axis=-1)




fig, axes = plt.subplots(2, 3, figsize=(8, 8))
ax = axes.ravel()
ax[0].axis('off')
ax[0].set_title("Original")
ax[0].imshow(image)

ax[1].axis('off')
ax[1].set_title("Rotated")
ax[1].imshow(rotated)

ax[2].axis('off')
ax[2].set_title("Rotated")
ax[2].imshow(rotated2)




ax[3].axis('off')
ax[3].set_title("Polar-Transformed Original")
ax[3].imshow(image_polar)

ax[4].axis('off')
ax[4].set_title(" Rotated 35 degree")
ax[4].imshow(rotated_polar)


ax[5].axis('off')
ax[5].set_title(" Rotated 60 degree")
ax[5].imshow(rotated_polar2)


#plt.show()

shifts, error, phasediff = phase_cross_correlation(
    image_polar, rotated_polar, normalization=None
)
print(f'Expected value for counterclockwise rotation in degrees: ' f'{angle}')
print(f'Recovered value for counterclockwise rotation: ' f'{shifts[0]}')




PolarTImages=[]
HistofPolor=[]

for image in images:
    image = img_as_float(image)
    image_polar = warp_polar(image, radius=radius, channel_axis=-1)
    PolarTImages.append(image_polar)
    n, bins, patches = plt.hist(image_polar.ravel(), 256, [0,1])

    bin_midpoints = (bins[:-1] + bins[1:]) / 2

    histo=[n,bin_midpoints]
    #print(histo)
    HistofPolor.append(histo)

PolarTImages = np.array(PolarTImages)
x_train, x_test= train_test_split( PolarTImages, test_size=0.2)

#HistofPolor= np.array(HistofPolor)
#PolarTImages = PolarTImages / 255.0 # required to load images
#x_train, x_test= train_test_split( PolarTImages, test_size=0.2)


##x_train, x_test= train_test_split(HistofPolor, test_size=0.2) #histo polar

#x_train, x_test= train_test_split( images, test_size=0.2)

f = open("predicat.txt", "w")
f.write(str(len(HistofPolor)))
f.write("\n-----------------------------\n")
f.write(str(PolarTImages[0].shape))
f.write("\n-----------------------------\n")
f.write(str(HistofPolor[0]))
f.write("\n-----------------------------\n")
  #f.write(str(predictions))
#f.write("\n-----------------------------\n")
#f.write(str(y_pred2))
#.write("\n-----------------------------\n")
#f.write(str(predictions[:][:][0]))
f.close()
#print(PolarTImages[1])

rowsize=180
colsize=180
def create_cnn_model_1(input_shape):
    inputs = Input(shape=input_shape)
    #x = layers.Conv2D(rowsize, (3, rowsize), activation='relu')(inputs)
    #x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3,3), activation='relu')(inputs)
    x = layers.MaxPooling2D(2,2 )(x)
    x = layers.BatchNormalization()(x)#
    x = layers.MaxPooling2D(2,2)(x)
     
    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D(2,2)(x)

    #x = layers.Conv1D(64, 3, activation='relu')(x)
   # x = layers.MaxPooling2D()(x)
   
    x = layers.Flatten()(x)
     
    x = layers.Dense(3, activation='relu')(inputs)
    return Model(inputs, x)


#input_shape = (2,255,3) # histo polar
input_shape = (360,110,3) #polar images
#input_shape = (240,240,3)# regular image

model = create_cnn_model_1(input_shape)
#binary_crossentropy,categorical_crossentropy,sparse_categorical_crossentropy

 

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'])



#train_generator = ImageDataGenerator().flow( x_train, 
 #     batch_size=32)

#history=model.fit(train_generator)
#y_train = to_categorical(x_train, num_classes=3)

history =model.fit(
  x_train,
  x_train,
  #validation_data=x_test.all(),
  epochs=3
)
 
 # Evaluate the model on the test set
#test_loss, test_acc = model.evaluate( PolarTImages)
#print(f"Test accuracy: {test_acc}")

acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#x = [1, 4]

#model.build(x)
#model.summary()

 