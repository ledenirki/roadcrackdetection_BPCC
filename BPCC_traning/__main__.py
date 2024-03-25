import cv2 
import numpy as np
import glob 
from skimage import io, morphology
from sklearn.model_selection import train_test_split

from functions import *


# 80*80 boyutunda çatlak içeren ve içermeyen görsellerin uzantılarının içeriye aktarılması

crack_paths= glob.glob(r"New_Dataset_80_80\Cracks\AllDatasets\*.png")  #Çatlak içeren veriler
non_crack_paths= glob.glob(r"New_Dataset_80_80\NonCracks\AllDatasets\*.jpg")   #Çatlak içermeyen veriler

# Verilerin içeri aktarılıp thresholdimage fonksiyonun uygulanması ve data olarak tek bir değişkende tutulması 
# [0,1] =çatlak değil [1,0]=çatlak

data= [] 
for path in crack_paths:  
    img= cv2.imread(path,0)  
    img_cm=thresholdimage(img)
    x= img_cm.reshape(img.shape[0]*img.shape[1],)         
    data.append((x))

for path in non_crack_paths:
    img= cv2.imread(path,0)  
    img_cm=thresholdimage(img)
    x= img_cm.reshape(img.shape[0]*img.shape[1],)      
    data.append((x))
    
# Datanın çıktı değerlerini düzenleyerek eğitim ve test olarak ana veri setinin ayrılması

a=np.zeros((len(data),len(data[0])))
for i in range(len(data)):
    a[i,:]=data[i]
data_n=a
a=np.tile([1,0],(7500,1))
b=np.tile([0,1],(7500,1))
Y=np.append(a,b,axis=0)
X=data_n
train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=42)

# Oluştulacak sinir ağının özelliklerinin belirlenmesi 

train_x = np.double(train_X)
test_x  = np.double(test_X)
train_y = np.double(train_y)
test_y  = np.double(test_y)
  
batchsize = 20

numinput = train_x.shape[1]
numhid_1 = 100
numoutput = train_y.shape[1]
m = np.size(train_x,0)

sizes = [numinput,numhid_1,numoutput] 

numbatches = m/batchsize

numepochs = 50 
iteration = 100
Threshold = 0.01
alph = 0.3
m_ratio = 0.3

# Ağırlık ve bias terimlerinin üretilmesi

nn_size                = sizes
nn_number_layer        = np.size(nn_size) - 1 

nn_ll                  = 2
nn_l                   = 0.00002
nn_l2                  = 0.000001

nn_W=[]
nn_b=[]

for i in range(1,nn_number_layer+1):
    
    nn_Wx = (np.random.rand(nn_size[i], nn_size[i - 1]) - 0.5) * 2 * 4 * np.sqrt(6 / (nn_size[i] + nn_size[i - 1]))  
    nn_W.append(nn_Wx)
    nn_bx = (np.random.rand(nn_size[i],1) - 0.5)*2
    nn_b.append(nn_bx)
    
Cost_all = np.zeros((numepochs,1))
Cost_index = np.zeros((numepochs,1))

iter_index = 1
subtask_number = sum(sizes[1:])
cost_last = 0

Wgrad= [[[0]],[[0]],[[0]]]
Delta= [[[0]],[[0]]]

# Elde edilen özellikler değerlerinin kullanılması ile öğrenme işleminin yapılması

for k in range(0,numepochs):    
    print(k)

    kk = np.random.permutation(range(m))
    
    x = train_x[kk[:iteration*batchsize],:]
    y = train_y[kk[:iteration*batchsize],:]
    
    for j in range(0,100):

        output=[]
        
        batch_x = x[(j)*batchsize:batchsize+batchsize*(j), :]
        batch_y = y[(j)*batchsize:batchsize+batchsize*(j),:]

        numpatches = np.size(batch_x.T,1)

        input_x = batch_x.T
        
        for i in range(0,nn_number_layer):
            output1= sigm(np.dot(nn_W[i], input_x) + np.tile(nn_b[i], (1, numpatches)))
            output.append(output1)
            input_x = output[i]
        
        Delta[nn_number_layer-1] = (output[nn_number_layer-1] - batch_y.T)*output[nn_number_layer-1]*(1 - output[nn_number_layer-1])
        for i in range(nn_number_layer-2,-1,-1):  
            Delta[i] = np.dot(nn_W[i+1].T,Delta[i+1])*output[i]*(1 - output[i])
        
        Wgrad[0] =  nn_l*nn_W[0] + np.dot(Delta[0],batch_x/numpatches)
        for i in range(1,nn_number_layer):
            Wgrad[i] = nn_l*nn_W[i] + np.dot(Delta[i],(output[i-1].T)/numpatches)
        
        for i in range(0,nn_number_layer):
            nn_W[i] = nn_W[i] - nn_ll*Wgrad[i]           
            nn_b[i] = nn_b[i] - np.expand_dims(nn_ll*np.sum(Delta[i],1)/numpatches, axis=1)        

    cost = cal_cost(x.T, y.T, nn_number_layer, nn_W, nn_b, nn_l2)
                   
    Cost_all[k,0] = cost
    
    nn=[]
    if abs(cost - cost_last) < Threshold/np.sqrt(k+1):
                
        [_, _, subtask_index_need] = cal_maturity(x.T, nn_number_layer, nn_W, nn_b, sizes, alph, m_ratio)
    
        nn.extend((nn_number_layer,nn_W,nn_b, nn_l2))
        globals()['nn_cc_' + str(iter_index)] = nn
        iter_index = iter_index + 1
        globals()['nn_cc_' + str(iter_index)] = saCCDE_all(x.T, y.T, globals()['nn_cc_' + str(iter_index-1)], sizes, subtask_index_need)

        nn = globals()['nn_cc_' + str(iter_index)]
        
        nn_number_layer= nn[0]
        nn_W= nn[1]
        nn_b= nn[2]
        nn_l2= nn[3]        
        Cost_all[k,0] = cal_cost(x.T, y.T, nn_number_layer,nn_W,nn_b, nn_l2)
            
    print(cost_last, cost)   
    cost_last = cost
    