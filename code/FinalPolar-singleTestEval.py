from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float
 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import cv2

import time

MODEL_PATH = 'polarTRmodel.h5'
IMAGE_PATH = '../DatasetRGB/Negative/gen_cvB_N1.png'
IMG_SIZE = (224, 224)  # Update to match your model's input size
CLASS_NAMES = ['Negative', 'Neutral', 'Positive']  # Update as needed

# 1. Load model
print("Loading model...")
model = load_model(MODEL_PATH)
print(f"Model input shape: {model.input_shape}")

# 2. Load and preprocess image
print("Loading image...")
img = image.load_img(IMAGE_PATH, target_size=IMG_SIZE)

img_array = image.img_to_array(img)
#image = np.array(img)
#image = image / 255.0  # Normalize if needed

image = img_as_float(img)
radius = 110 # workeed for input polar tested
angle = 35
image_polar = warp_polar(image, radius=radius, channel_axis=-1)


n, bins, patches = plt.hist(image_polar.ravel(), 256, [0,1])
bin_midpoints = (bins[:-1] + bins[1:]) / 2
r = n  # The radius is the height of the histogram bin
theta = bin_midpoints * 2 * np.pi 
histoT=[r,theta]

Rtemp_R=np.zeros_like(r)
Rtemp_theta=np.zeros_like(theta)

temp_R = np.copy(r)  # Create a copy to preserve original data
temp_theta = np.copy(theta)

angle_min = np.deg2rad(50)   # Example: 30 degrees in radians
angle_max = np.deg2rad(130)
angles = np.linspace(angle_min, angle_max, 100)
Afiltered_indices = (theta >= angle_min) & (theta <= angle_max)

Rtemp_R[~((theta >= angle_min) & (theta <= angle_max))] = 0    
Rtemp_theta[~((theta >= angle_min) & (theta <= angle_max))] = 0 

    #Rtemp_R = theta[Afiltered_indices]
    #Rtemp_theta = r[Afiltered_indices]

histoR=[Rtemp_R,Rtemp_theta]
HistofPolor=[]
HistofPolor.append(histoR)
HistofPolor=np.array(HistofPolor)

start_time = time.time()
print("Making prediction...")
predictions = model.predict(HistofPolor, verbose=0)

end_time = time.time()
 
duration = end_time - start_time
 
print("\n" + "="*50)
print("PREDICTION RESULTS")
print("="*50)

if predictions.shape[-1] == 1:
    
    prob = predictions[0][0]
    print(f"Probability: {prob:.4f}")
    print(f"Prediction: {'Positive' if prob > 0.5 else 'Negative'}")
    print(f"respone time={duration:.4f} seconds")
else:
    # Multi-class
    pred_class = np.argmax(predictions[0])
    confidence = predictions[0][pred_class]
    
    if CLASS_NAMES and pred_class < len(CLASS_NAMES):
        class_name = CLASS_NAMES[pred_class]
    else:
        class_name = f"Class {pred_class}"
    
    print(f"Predicted class: {class_name}")
    print(f"respone time={duration:.4f} seconds")
   # print(f"Confidence: {confidence:.4f}")
    
    # Show top 3 predictions
   # top_indices = np.argsort(predictions[0])[-3:][::-1]
    #print("\nTop 3 predictions:")
    #for idx in top_indices:
     #   if CLASS_NAMES and idx < len(CLASS_NAMES):
       #     name = CLASS_NAMES[idx]
       # else:
       #     name = f"Class {idx}"
       # print(f"  {name}: {predictions[0][idx]:.4f}")

 
