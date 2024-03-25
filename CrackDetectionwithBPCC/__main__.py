import pickle
import cv2 
import numpy as np
from skimage import morphology
from PIL import Image

from functions import crack_detection

#Çatlak tespiti yapmak istenen görselin uzantısının girilmesi
path= r"Aaaaaa.jpg"

#Yönteme başlamadan hemen önceki zaman değeri
start_time = time.time()

#Sırasıyla, çatlak olmayan nesnelerden arındırılmış binary görsel, çatlak alanlarının konumlarının 
#ve çatlak bölgesi tespit edilmiş görselin elde edilmesi
image1, cracknodes, image2= crack_detection(path)

#Yöntemin tespit için harcadığı süre
passed_time = time.time() - start_time

#Elde edilen görsellerin gösterilmesi
cv2.imshow("Image 1",image)
cv2.imshow("Image 2",image1)