import glob 
from crack_detection_new_v1 import crack_detection
import cv2
import time
import pickle
from math import floor
import numpy as np

#Extension of manually specified binary crack image
groundtruth_path= r"Aaaaaa.jpg"
#Extension of the coloured image for the achievement test
rgb_image_path= r"Aaaaaa.jpg"

#Import of BPCC weight parameters
ws_file= open(r"nn_W.pkl","rb")
nn_W = pickle.load(ws_file)
ws_file.close()

bs_file= open(r"nn_b.pkl","rb")
nn_b = pickle.load(bs_file)
bs_file.close()


TP=0    #True-Positive
FP=0    #False-Positive
FN=0    #False-Negative

#Transferring the image to be determined success to the inside in gray-scale
img= cv2.imread(original_paths[i],0)    

#Transferring the visual to be used for success determination into the interior in rgb image
img_r= cv2.imread(original_paths[i]) 

#Using of the crack detection method
image,cracknodes, image1= crack_detection(img,img_r,nn_W,nn_b)

#Importing and binarising the ground truth crack image
real_image= cv2.imread(real_paths[i],0)
_, real_image=cv2.threshold(real_image, 100, 255,cv2.THRESH_BINARY)
img_Real=real_image.copy()

#Calculation of TP and FP values in the crack image
for i in range(len(cracknodes)):
    area=real_image[cracknodes[i][0]:cracknodes[i][1], cracknodes[i][2]:cracknodes[i][3]]       
    img_Real[cracknodes[i][0]:cracknodes[i][1], cracknodes[i][2]:cracknodes[i][3]]=0       
    if np.count_nonzero(area)>0:
        TP=TP+1
    else:
        FP=FP+1

#Calculation of the FN value in the crack image
_, contours, _ = cv2.findContours(img_Real, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    ratio=w*h/(80*80)
    number=floor(ratio)+1
    FN=FN+number 
    
precision= TP/(TP+FP)  
recall= TP/(TP+FN)
f1= 2*(precision*recall)/(precision+recall)  
print("Precision: %"+ precision*100, "Recall: %"+ recall*100, "F1-Score: %"+ f1*100) 
