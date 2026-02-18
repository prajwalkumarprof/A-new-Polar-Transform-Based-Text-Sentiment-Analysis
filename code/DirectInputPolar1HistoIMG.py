

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

#image_folder = '../Datasetww/'
#image_folder = '../DatasetwithR/'
#image_folder = '../DatasetRGB/'

image_folder = '../DatasetpolarPlotin/'

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





PolarTImages=[]
HistofPolor=[]
HistofPolorImg=[]
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.axis('off')

def fig2img(fig): 
    buf = io.BytesIO() 
    fig.savefig(buf) 
    buf.seek(0) 
    img = Image.open(buf) 
    return img 

fig3 = plt.figure()
ax1 = fig3.add_subplot(111,  polar=True)
for image in images:
    image = img_as_float(image)
    image_polar = warp_polar(image, radius=radius, channel_axis=-1)
    PolarTImages.append(image_polar)

    n, bins, patches = plt.hist(image_polar.ravel(), 256, [0,1])
    bin_midpoints = (bins[:-1] + bins[1:]) / 2
    r = n  # The radius is the height of the histogram bin
    theta = bin_midpoints * 2 * np.pi 
    histo=[r,theta]
    HistofPolor.append(histo)    

    #row =img_as_float(row) 
    #row = fig2img(fig)
    #print(histo)
   
    
    ax1.scatter(theta, r,  c='blue', s=10, cmap='hsv', label='Bin', alpha=0.75)
    ax1.legend()
    ax1.set_rmax(500)
    ax1.axis('off')
    #plt.show()
   
    HistofPolorImg.append(fig2img(fig3))


#PolarTImages = np.array(PolarTImages)
#x_train, x_test= train_test_split( images, test_size=0.2)

#HistofPolor= np.array(HistofPolor)
#x_train, x_test= train_test_split(HistofPolor, test_size=0.2) #histo polar

#PolarTImages = PolarTImages / 255.0 # required to load images
#x_train, x_test= train_test_split( PolarTImages, test_size=0.2)

#HistofPolorImg= np.array(HistofPolorImg)
#x_train, x_test= train_test_split(HistofPolorImg, test_size=0.2) #histo polar


#original images
x_train, x_test= train_test_split( images, test_size=0.2)

f = open("predicat.txt", "w")
f.write(str(len(HistofPolor)))
f.write("\n-----------------------------\n")
f.write(str(HistofPolor[0]))
f.write("\n-----------------------------\n")
#f.write(str(HistofPolor[0]))

#f.write(str(predictions[:][:][0]))
f.close()
#print(PolarTImages[1])

rowsize=180
colsize=180
 
 
def create_cnn_model_O(input_shape):
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(256, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3,3), activation='ReLU')(x)
    x = layers.MaxPooling2D((3,3))(x)
    #x = layers.Conv2D(64, (3,3), activation='ReLU')(x)
    #x = layers.MaxPooling2D((3,3))(x)
    x = layers.BatchNormalization()(x)#
    x = layers.Dense(64 ,activation='ReLU')(x)  
   # x = layers.Flatten()(x)
   # x = layers.Dense(256 ,activation='ReLU')(inputs)
    x =  layers.Dense(3 ,activation='softmax')(inputs)
    #x = layers.Dense(3, activation='ReLU')(x)
    #outputs = layers.Dense(10, activation='softmax')(x)
    #outputs = layers.Dense(10, activation='softmax')(x)
    return Model(inputs,outputs=x )
# working with abole cnn create_cnn_model_O


#input_shape = (360,110,3) #polar images

input_shape = (240,240,3)# regular image
model = create_cnn_model_O(input_shape)




#binary_crossentropy,categorical_crossentropy,sparse_categorical_crossentropy


model1 = tf.keras.Sequential([
  #  abalone_features,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    
    tf.keras.layers.Dense(256)  # Assuming 3 classes for the output layer
])

#model=model1
#sparse_categorical_crossentropy

model.compile(
  optimizer='SGD',
  loss='mean_absolute_error',
  metrics=['accuracy'])



#train_generator = ImageDataGenerator().flow( x_train, 
 #     batch_size=32)

#history=model.fit(train_generator)
#y_train = to_categorical(x_train, num_classes=3)

history =model.fit(
  x_train,
  x_train,
  validation_data=x_test.all(),
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

 