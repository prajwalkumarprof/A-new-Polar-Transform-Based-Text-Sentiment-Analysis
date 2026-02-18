import numpy as np
 
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import csv
# Function to calculate Chi-distance
import statistics

def chi2_distance(histA, histB, eps = 1e-10):
	# compute the chi-squared distance
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
		for (a, b) in zip(histA, histB)])
	# return the chi-squared distance
	return d
 
# main function
if __name__== "__main__":
    a = [1, 0, 0, 5, 45, 23]
    b = [67, 90, 0, 79, 24, 98]
 
  #  result = chi2_distance(a, b)
   # print("The Chi-square distance is :", result)




def featureExtractionHISTO(filename):
        image = cv2.imread(filename['file'])
        img_gray = cv2. cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(filename)
        img_normalizedgray = cv2.normalize(img_gray, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
       # print(img_gray)
        blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)
        dst = cv2.Canny(img_gray, 50, 200, None, 3)

        b,g,r = cv2.split(image)
        img_normalized_b = cv2.normalize(b,   0, 256, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_normalized_r = cv2.normalize(r,   0, 256, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_normalized_g = cv2.normalize(g,   0, 256, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        fname=filename['name']
        #classname=filename['class']
        classname=""
        #cv2.imwrite('testresult/threshimage'+fname , thresh) 
       # cv2.imwrite('testresult/grayimage'+fname , img_normalized_r) 
       # cv2.imwrite('testresult/cannyimage'+fname , dst) 
         
        
        plt.imshow(img_gray)
        plt.axis('off')
        plt.savefig( Dpath+classname+"/"+"gray"+fname)
        plt.clf()

         #axisgray=plt.hist(img_normalizedgray.ravel(),256,[0,1])
       # axisgray=plt.hist(img_normalized_r.ravel(),256,[0,1])
       # plt.title('GRAY HISTOGRAM ')
       # plt.xlabel("value")
       # plt.ylabel("Frequency")
        #plt.savefig( filename+"Histo-Zgray.png")
        #ax2.imshow(axisgray)
       # plt.show()
       # plt.clf()
 
        plt.title("Histogram of AdaptiveThreshold")
        axisB=plt.hist(thresh.ravel(),256,[0,1]) 
        plt.xlabel("value")
        plt.ylabel("Frequency")
        plt.savefig(Dpath+classname+"/"+fname+"G.png") 
        plt.clf()

     
       # plt.subplot(3, 1, 3)
        plt.title("Histogram of Red  ")
        axisg=plt.hist(img_normalized_r.ravel(),256,[0,1]) 
        plt.savefig(Dpath+"/"+fname+"hsitogramofred.png");
        plt.clf()
        



                



 
        linedata=[]
        return linedata

def RGBGEN_image(path,imageslist):
 
    for filename in imageslist: 
      image = cv2.imread(filename['file'])
      img_gray = cv2. cvtColor(image, cv2.COLOR_BGR2GRAY)
      b,g,r = cv2.split(image)
      img_normalized_b = cv2.normalize(b,   0, 256, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
      img_normalized_r = cv2.normalize(r,   0, 256, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
      img_normalized_g = cv2.normalize(g,   0, 256, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

      fname=filename['name']
      classname=filename['class']  
      cv2.imwrite(Dpath+classname+"/"+"Gray_"+fname , img_gray) 
      
      cv2.imwrite(Dpath+classname+"/"+"R_"+fname , r) 
      cv2.imwrite(Dpath+classname+"/"+"G_"+fname , g) 
      cv2.imwrite(Dpath+classname+"/"+"B_"+fname , b)   
      #plt.imshow(r)
      #plt.axis('off')
      #plt.savefig( Dpath+classname+"/"+"gen_gray_"+fname)
      #plt.clf()

      #plt.imshow(img_normalized_r)
      #plt.axis('off')
      #plt.savefig( Dpath+classname+"/"+"gen_R_"+fname)
      #plt.clf()

     # plt.imshow(img_normalized_g)
     # plt.axis('off')
     # plt.savefig( Dpath+classname+"/"+"gen_G_"+fname)
     # plt.clf()

      plt.imshow(img_gray)
      plt.axis('off')
      plt.savefig( Dpath+classname+"/"+"gen_pgray_"+fname)
      plt.clf()
      #linedata=featureExtractionHISTO(filename)
     
def load_from_folderclass():
  for pathb in Directorylist:
    #  print(pathb)
    
    # if os.path.isdir(pathb):
      for filename in os.listdir(path+pathb): # FOLDER
          # print(path+pathb)
            if filename.endswith(".png"):
              #imageslist.append(path+pathb+"/"+filename)
              imageslist.append({'file': path+pathb+"/"+filename,'name':filename, 'class': pathb} )
            #  classlist.append(pathb )
              onlyFilename.append(filename)

i=0
path="tesdocimages/"
#path="testplotinput/"
#path="../Datasetww/"
#path="./fruit-class4/"
#path="./leaf-class5/"
Dpath="TRInputRGB/"       



Directorylist = []

 
#path="../testinput/"


for pathb in os.listdir(path):
    Directorylist.append(pathb)

Directorylist.sort(key=str.lower)

onlyFilename=[]
imageslist = []

#classlist=[]
# ONLY FOR IMAGE IN GIVEN FOLDER 

for filename in  os.listdir(path):
    pathb=""
    print(pathb)     
    if filename.endswith(".png") or filename.endswith(".jpg"):
              #imageslist.append(path+pathb+"/"+filename)
      imageslist.append({'file': path+pathb+"/"+filename,'name':filename, 'class': pathb} )
            #  classlist.append(pathb )
      onlyFilename.append(filename)




RGBGEN_image(path,imageslist)
