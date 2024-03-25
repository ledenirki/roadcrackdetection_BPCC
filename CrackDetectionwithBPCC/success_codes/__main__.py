import glob 
from crack_detection_new_v1 import crack_detection
import cv2
import time
import pickle
from math import floor
import numpy as np

#Manuel olarak belirlenmiş binary çatlak görselinin uzantısı
groundtruth_path= r"Aaaaaa.jpg"
#Başarı testi yapılacak renkli görselin uzantısı
rgb_image_path= r"Aaaaaa.jpg"

#GYBE ağırlık parametrelerinin içeriye aktarılması 
ws_file= open(r"nn_W.pkl","rb")
nn_W = pickle.load(ws_file)
ws_file.close()

bs_file= open(r"nn_b.pkl","rb")
nn_b = pickle.load(bs_file)
bs_file.close()


TP=0    #True-Positive
FP=0    #False-Positive
FN=0    #False-Negative

#Başarı tespiti yapılacak görselin gri tonda içeriye aktarılması
img= cv2.imread(original_paths[i],0)    

#Başarı tespiti yapılacak görselin renkli tonda içeriye aktarılması
img_r= cv2.imread(original_paths[i]) 

#Çatlak tespiti yönteminin uygulanması
image,cracknodes, image1= crack_detection(img,img_r,nn_W,nn_b)

#Ground truth çatlak görselinin içeriye aktarılması ve binary hale getirilmesi
real_image= cv2.imread(real_paths[i],0)
_, real_image=cv2.threshold(real_image, 100, 255,cv2.THRESH_BINARY)
img_Real=real_image.copy()

#Çatlak görselindeki TP ve FP değerlerinin hesaplanması
for i in range(len(cracknodes)):
    area=real_image[cracknodes[i][0]:cracknodes[i][1], cracknodes[i][2]:cracknodes[i][3]]       
    img_Real[cracknodes[i][0]:cracknodes[i][1], cracknodes[i][2]:cracknodes[i][3]]=0       
    if np.count_nonzero(area)>0:
        TP=TP+1
    else:
        FP=FP+1

#Çatlak görselindeki FN değerinin hesaplanması
_, contours, _ = cv2.findContours(img_Real, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    ratio=w*h/(80*80)
    number=floor(ratio)+1
    FN=FN+number 
    
precision= TP/(TP+FP)  
recall= TP/(TP+FN)
f1= 2*(precision*recall)/(precision+recall)  
print("Kesinlik: %"+ precision*100, "Duyarlılık: %"+ recall*100, "F1-Score: %"+ f1*100) 