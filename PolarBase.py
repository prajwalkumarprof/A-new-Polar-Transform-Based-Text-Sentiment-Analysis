


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



image_folder = '../Dataset333/BOTH/'

Directorylist = []
PolarTImages=[]
HistofPolor=[]
HistofPolorImg=[]

radius = 110  
angle = 35

def loaddataset():
    print("load dataset :")
    
 
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
   
    histpolar = Extractfeaures(images)
    print("load completed :")
    return histpolar
   

def getTRdata(min,max,theta,r):
        temp_R = np.copy(r) 
        temp_theta = np.copy(theta)

        angle_min = np.deg2rad(min)   
        angle_max = np.deg2rad(max)
        angles = np.linspace(angle_min, angle_max, 100)
        Afiltered_indices = (theta >= angle_min) & (theta <= angle_max)

        temp_R[~((theta >= angle_min) & (theta <= angle_max))] = 0   
        temp_theta[~((theta >= angle_min) & (theta <= angle_max))] = 0 
        histT=[temp_R,temp_theta]

        return histT

def Extractfeaures(images):
    print("Extractfeaures :")

    for image in images:
        image = img_as_float(image)
        image_polar = warp_polar(image, radius=radius, channel_axis=-1)
        PolarTImages.append(image_polar)


        n, bins, patches = plt.hist(image_polar.ravel(), 256, [0,1])
        bin_midpoints = (bins[:-1] + bins[1:]) / 2
        r = n  
        theta = bin_midpoints * 2 * np.pi 
        Rtemp_R=np.zeros_like(r)
        Rtemp_theta=np.zeros_like(theta)

        temp_R = np.copy(r) 
        temp_theta = np.copy(theta)

        angle_min = np.deg2rad(50)   
        angle_max = np.deg2rad(130)
        angles = np.linspace(angle_min, angle_max, 100)
        Afiltered_indices = (theta >= angle_min) & (theta <= angle_max)

        Rtemp_R[~((theta >= angle_min) & (theta <= angle_max))] = 0    
        Rtemp_theta[~((theta >= angle_min) & (theta <= angle_max))] = 0 

        histoR=[Rtemp_R,Rtemp_theta]

        radius_min = 350   
        radius_max = 500   
            
        temp_R[~((r >= radius_min) & (r <= radius_max))] = 0    
        temp_theta[~((r >= radius_min) & (r <= radius_max))] = 0 
      
        histoT=[temp_R,temp_theta]  
 
        

        HistofPolor.append(histoR)
        #print("...feaures extracting...")
    
    return HistofPolor