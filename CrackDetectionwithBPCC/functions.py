import cv2 
import numpy as np
from skimage import morphology
import pickle
from PIL import Image
 

def connectivitymaps(img):
    threshold= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 135, 18)
    
    if np.count_nonzero(threshold)>6000 or np.count_nonzero(threshold)<400:
        threshold= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 135, 5)
    
    erosion_w= cv2.bitwise_not(threshold)
    arr = erosion_w > 0
    cleaned = morphology.remove_small_objects(arr, min_size=50)
    cleaned1 = morphology.remove_small_holes(cleaned, 150)            
    result= 1*cleaned1                   
    return result 
    
def sigm(P):    
    X = 1/(1+np.exp(-P))
    return X

def detect_crack(img,nn_W,nn_b):    
    for i in range(0,2): 
        img= sigm(np.dot(nn_W[i],img) + nn_b[i])  
    
    y_pred= img.T 
    y_pred=np.round(y_pred)
    if y_pred[0][0]==1 and y_pred[0][1]==0:   
        result=True
    else:
        result=False
    
    return result

def removed_small_object(img,cleaned1,nn_W,nn_b):    
    areaimg= (img.shape[0])*(img.shape[1])

    cleaned2=cleaned1.astype(np.uint8)*255
    contours, _ = cv2.findContours(cleaned2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        ax=((box[0][0]-box[1][0])**2+(box[0][1]-box[1][1])**2)**0.5
        ay=((box[0][0]-box[3][0])**2+(box[0][1]-box[3][1])**2)**0.5
        area2= (1/2)*(box[0][0]*box[1][1]+box[1][0]*box[2][1]+box[2][0]*box[3][1]+box[3][0]*box[0][1]-box[1][0]*box[0][1]-box[2][0]*box[1][1]-box[3][0]*box[2][1]-box[0][0]*box[3][1])
        rate= min(ax,ay)/max(ax,ay)
        if area2<(areaimg/512) and rate>0.25:
            cleaned2[y:y+h,x:x+w]=0    
                         
        return cleaned2
    
def limited_1(x,y,w,h,cleaned5,image,img,nn_W,nn_b,cracknodes):
    
    if w<=80 and h<=80:   
        box= [int(y+h/2-40),int(y+h/2+40),int(x+w/2-40),int(x+w/2+40)]
        if x+w/2-40<0:
            box[2],box[3]=0,80
        elif x+w/2+40>=cleaned5.shape[1]:
            box[2],box[3]= cleaned5.shape[1]-80, cleaned5.shape[1]
        if y+h/2-40<0:
            box[0],box[1]=0,80
        elif y+h/2+40>=cleaned5.shape[0]:
            box[0],box[1]= cleaned5.shape[0]-80, cleaned5.shape[0] 
            
        part_of_image=img[box[0]:box[1],box[2]:box[3]]
        binary_image=connectivitymaps(part_of_image)
        line_image= binary_image.reshape(binary_image.shape[0]*binary_image.shape[1],1)
        result= detect_crack(line_image,nn_W,nn_b) 
        if result==True:
            cleaned_part=cleaned5[box[0]:box[1],box[2]:box[3]]
            _, contours_part, _ = cv2.findContours(cleaned_part, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours_part)<2:
                cv2.rectangle(image,(box[2],box[0]),(box[3],box[1]),(255,255,255),-1)
                nodes=[box[0],box[1],box[2],box[3]]
                cracknodes.append(nodes)
    
    elif w>80 and h<=80:
        box= [int(y+h/2-40),int(y+h/2+40),x,x+w]
        abc=int(w/80)
        if int(w/80)/(w/80) != 1:
            fark=(int(w/80)+1)*80-w
            fark1,fark2=x,cleaned5.shape[1]-x-w
            if fark1>=fark/2 and fark2>=fark/2:
                x=x-int(fark/2)
            elif fark1<=fark/2 and fark2>=fark/2:
                x=0
            elif fark1>=fark/2 and fark2<=fark/2:
                x=cleaned5.shape[1]-(int(w/80)+1)*80
            abc= int(w/80)+1
        if y+h/2-40<0:
            box[0],box[1]=0,80
        elif y+h/2+40>=cleaned5.shape[0]:
            box[0],box[1]= cleaned5.shape[0]-80, cleaned5.shape[0]
        for i in range(abc): 
            box1= [box[0],box[1],int(x+i*80),int(x+(i+1)*80)]
            part_of_image=img[box1[0]:box1[1],box1[2]:box1[3]]
            binary_image=connectivitymaps(part_of_image)
            line_image= binary_image.reshape(binary_image.shape[0]*binary_image.shape[1],1)
            result= detect_crack(line_image,nn_W,nn_b) 
            if result==True:
                cv2.rectangle(image,(box1[2],box1[0]),(box1[3],box1[1]),(255,255,255),-1)
                nodes=[box1[0],box1[1],box1[2],box1[3]]
                cracknodes.append(nodes)
            else:
                a=[(20,0),(-20,0),(0,20),(0,-20)]
                for i in range(4):
                    box2= box1[0]+a[i][0],box1[1]+a[i][0],box1[2]+a[i][1],box1[3]+a[i][1]
                    if min(box2)>=0 and box2[1]<=img.shape[0] and box2[3]<=img.shape[1]:
                        part_of_image=img[box2[0]:box2[1],box2[2]:box2[3]]
                        binary_image=connectivitymaps(part_of_image)
                        line_image= binary_image.reshape(binary_image.shape[0]*binary_image.shape[1],1)
                        result= detect_crack(line_image,nn_W,nn_b) 
                        if result==True:
                            cv2.rectangle(image,(box2[2],box2[0]),(box2[3],box2[1]),(255,255,255),-1)
                            nodes=[box2[0],box2[1],box2[2],box2[3]]
                            cracknodes.append(nodes)
                            break
                        
    elif w<=80 and h>80:
        box= [y,y+h,int(x+w/2-40),int(x+w/2+40)]
        abc=int(h/80)
        if int(h/80)/(h/80) != 1:
            fark=(int(h/80)+1)*80-h
            fark1,fark2=y,cleaned5.shape[0]-y-h
            if fark1>=fark/2 and fark2>=fark/2:
                y=y-int(fark/2)
            elif fark1<=fark/2 and fark2>=fark/2:
                y=0
            elif fark1>=fark/2 and fark2<=fark/2:
                y=cleaned5.shape[0]-(int(h/80)+1)*80
            abc= int(h/80)+1
        if x+w/2-40<0:
            box[2],box[3]=0,80
        elif x+w/2+40>=cleaned5.shape[1]:
            box[2],box[3]= cleaned5.shape[1]-80, cleaned5.shape[1]
        for i in range(abc): 
            box1= [int(y+i*80),int(y+(i+1)*80),box[2],box[3]]
            part_of_image=img[box1[0]:box1[1],box1[2]:box1[3]]
            binary_image=connectivitymaps(part_of_image)
            line_image= binary_image.reshape(binary_image.shape[0]*binary_image.shape[1],1)
            result= detect_crack(line_image,nn_W,nn_b) 
            if result==True:
                cv2.rectangle(image,(box1[2],box1[0]),(box1[3],box1[1]),(255,255,255),-1)
                nodes=[box1[0],box1[1],box1[2],box1[3]]
                cracknodes.append(nodes)
            else:
                a=[(20,0),(-20,0),(0,20),(0,-20)]
                for i in range(4):
                    box2= box1[0]+a[i][0],box1[1]+a[i][0],box1[2]+a[i][1],box1[3]+a[i][1]
                    if min(box2)>=0 and box2[1]<=img.shape[0] and box2[3]<=img.shape[1]:
                        part_of_image=img[box2[0]:box2[1],box2[2]:box2[3]]
                        binary_image=connectivitymaps(part_of_image)
                        line_image= binary_image.reshape(binary_image.shape[0]*binary_image.shape[1],1)
                        result= detect_crack(line_image,nn_W,nn_b) 
                        if result==True:
                            cv2.rectangle(image,(box2[2],box2[0]),(box2[3],box2[1]),(255,255,255),-1)
                            nodes=[box2[0],box2[1],box2[2],box2[3]]
                            cracknodes.append(nodes)
                            break                        
    return image

def crack_detection(path):
    
    ws_file= open(r"nn_W.pkl","rb")
    nn_W = pickle.load(ws_file)
    ws_file.close()
    
    bs_file= open(r"nn_b.pkl","rb")
    nn_b = pickle.load(bs_file)
    bs_file.close()

    img= cv2.imread(path,0)    
    img_r= cv2.imread(path)  
    
    threshold= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 135, 0.233*(img.shape[0]*img.shape[1])**0.33)
    
    erosion_w= cv2.bitwise_not(threshold)
    arr = erosion_w > 0
    
    cleaned = morphology.remove_small_objects(arr, min_size=50)
    cleaned1 = morphology.remove_small_holes(cleaned, 150)   
    
    cleaned3=removed_small_object(img,cleaned1,nn_W,nn_b)
    cleaned5=cleaned3.copy()
    
    image= np.zeros((img.shape[0],img.shape[1]))
    areaimg= (img.shape[0])*(img.shape[1])
           
    _, contours, _ = cv2.findContours(cleaned5, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        list1 = c.tolist()
        xlist=[]
        ylist=[]
        for i in range(len(c)):
            xlist= np.append(xlist,list1[i][0][0])
            ylist= np.append(ylist,list1[i][0][1])
        
        min_x = list1[xlist.argmin(0)][0]
        max_x = list1[xlist.argmax(0)][0]
        
        min_y = list1[ylist.argmin(0)][0]
        max_y = list1[ylist.argmax(0)][0]
        
        box=[min_x,max_x,min_y,max_y]
        for i in range(4):
            if box[i][1]>=10 and box[i][1]<img.shape[0]-10 and box[i][0]>=10 and box[i][0]+10<img.shape[1]-10:
                part= cleaned5[box[i][1]-10:box[i][1]+10,box[i][0]-10:box[i][0]+10]
                _, contours1, _ = cv2.findContours(part, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours1)==2:
                    array_of_tuples = map(tuple, contours1[0][0])
                    start=tuple(array_of_tuples)
                    startx,starty = start[0]
                    
                    array_of_tuples = map(tuple, contours1[1][-1])
                    end = tuple(array_of_tuples)
                    endx,endy = end[0]          
                    cv2.line(cleaned5,(startx+box[i][0]-10,starty+box[i][1]-10),(endx+box[i][0]-10,endy+box[i][1]-10),[255,255,255],2)
    
                    
    cleaned6=cleaned5.copy()
    
    arr = cleaned6 > 0                 
    cleaned6 = morphology.remove_small_objects(arr, min_size=50)
    cleaned6 = morphology.remove_small_holes(cleaned6, 150)*255
    cleaned6=cleaned6.astype(np.uint8)
    
    _, contours, _ = cv2.findContours(cleaned6, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        ax=((box[0][0]-box[1][0])**2+(box[0][1]-box[1][1])**2)**0.5
        ay=((box[0][0]-box[3][0])**2+(box[0][1]-box[3][1])**2)**0.5
        area2= (1/2)*(box[0][0]*box[1][1]+box[1][0]*box[2][1]+box[2][0]*box[3][1]+box[3][0]*box[0][1]-box[1][0]*box[0][1]-box[2][0]*box[1][1]-box[3][0]*box[2][1]-box[0][0]*box[3][1])
        rate= min(ax,ay)/max(ax,ay)
        if area2<(areaimg/615):
            cleaned6[y:y+h,x:x+w]=0   
        elif (areaimg/615)<area2<(areaimg/307) and rate>0.4:
            cleaned6[y:y+h,x:x+w]=0 
    
    cracknodes=[]        
    _, contours4, _ = cv2.findContours(cleaned6, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours4: 
        (x, y, w, h) = cv2.boundingRect(c)
        cleaned_s= np.zeros((img.shape[0],img.shape[1]))
        cleaned_s=cleaned_s.astype(np.uint8)
        finished=False
        cleaned_s[y:y+h,x:x+w]=cleaned6[y:y+h,x:x+w]
        while finished==False: 
            (x, y, w, h) = cv2.boundingRect(c)
            if w>80 and h>80:
                hull = cv2.convexHull(c,returnPoints = False)
                defects = cv2.convexityDefects(c,hull)
                far_list=[]
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    far = tuple(c[f][0])
                    far_list.append(far)
                    
                list_i=[]
                box_list =[] 
                for far in far_list:
                    xfar,yfar= far 
                    if xfar<40:
                        xfar=40
                    elif xfar>cleaned_s.shape[1]-40:
                        xfar=cleaned_s.shape[1]-40
                    if yfar<40:
                        yfar=40
                    elif yfar>cleaned_s.shape[0]-40:
                        yfar=cleaned_s.shape[0]-40   
                        
                    #dikey                 
                    i_d=0
                    for far1 in far_list:
                        if xfar-40<=far1[0]<xfar+40:
                            i_d+=1                
                    #yatay
                    i_y=0
                    for far2 in far_list:
                        if yfar-40<=far2[1]<yfar+40:
                            i_y+=1
                    if i_d>i_y:
                        boxdy=[xfar-40,y,xfar+40,y+h]
                        list_i.append(i_d)
    
                    else:
                        boxdy=[x,yfar-40,x+w,yfar+40]
                        list_i.append(i_y) 
                        
                    box_list.append(boxdy)
            
                value_max, num_max = np.array(list_i).max(0),np.array(list_i).argmax(0)    
                x1,y1,x2,y2=box_list[num_max]
                cv2.rectangle(cleaned_s,(x1,y1),(x2,y2),(0,0,0),-1)
                image=limited_1(x1,y1,x2-x1,y2-y1,cleaned_s,image,img,nn_W,nn_b,cracknodes)
       
                arr = cleaned_s > 0
                cleaned_s = morphology.remove_small_objects(arr, min_size=30)
                cleaned_s=cleaned_s.astype(np.uint8)*255
                _, contoursdy, _ = cv2.findContours(cleaned_s,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                if len(contoursdy)==0:
                    finished=True
                elif len(contoursdy)==1:
                    c=contoursdy[0]
                else:
                    breaak=True
                    for cnt in contoursdy: 
                        (x, y, w, h) = cv2.boundingRect(cnt)
                        if w>80 and h>80:
                            breaak= False
                            c=cnt
                        else:
                            image=limited_1(x,y,w,h,cleaned_s,image,img,nn_W,nn_b,cracknodes)
                            
                    if breaak==True:
                        finished=True
        
            else:
                image=limited_1(x,y,w,h,cleaned_s,image,img,nn_W,nn_b,cracknodes)
                finished=True
                
    image=image.astype(np.uint8)
    image_r= cv2.bitwise_not(image)
    edge=cv2.Canny(image,100,255)
        
    kernel = np.ones((2,2), np.uint8)
    img_dilation = cv2.dilate(edge, kernel, iterations=1)
    
    edge_r= cv2.bitwise_not(img_dilation)
    
    red_img  = np.full((img_r.shape[0],img_r.shape[1],img_r.shape[2]), (0,0,255), np.uint8)
    white_img  = np.full((img_r.shape[0],img_r.shape[1],img_r.shape[2]), (0,0,0), np.uint8)
    fused_img  = cv2.addWeighted(img_r, 0.8, red_img, 0.2, 0)
    
    crack_part= cv2.bitwise_and(fused_img,fused_img,mask=image)
    non_crack_part= cv2.bitwise_and(img_r,img_r,mask=image_r)
    all_picture_v1=crack_part+non_crack_part
    
    all_picture_v2=cv2.bitwise_and(all_picture_v1,all_picture_v1,mask=edge_r) 
          
    return cleaned6, cracknodes, all_picture_v2
