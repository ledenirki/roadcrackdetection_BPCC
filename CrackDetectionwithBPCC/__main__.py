import pickle
import cv2 
import numpy as np
from skimage import morphology
from PIL import Image

from functions import crack_detection

#Entering the extension of the image for crack detection
path= r"Aaaaaa.jpg"

#Time value immediately before starting the method
start_time = time.time()

#The binary image, respectively, is a binary visualisation of the positions of the cracked areas and obtaining the image with the crack area detected
image1, cracknodes, image2= crack_detection(path)

#Time taken by the method for detection
passed_time = time.time() - start_time

#Showing the visuals obtained
cv2.imshow("Image 1",image)
cv2.imshow("Image 2",image1)
