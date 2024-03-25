import cv2 
import numpy as np
from skimage import morphology

def thresholdimage(img):
    threshold= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 135, 18)
    
    if np.count_nonzero(threshold)>6000 or np.count_nonzero(threshold)<400:
        threshold= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 135, 5)
    
    erosion_w= cv2.bitwise_not(threshold)
    arr = erosion_w > 0
    cleaned = morphology.remove_small_objects(arr, min_size=150)
    cleaned1 = morphology.remove_small_holes(cleaned, 150)
    result= 1*cleaned1
    return result 

def sigm(P):
    X = 1/(1+np.exp(-P))
    return X

def g_f(T, alph):
    w = np.sum(np.logical_or((T<alph),T>(1-alph)),1)
    return w

def cal_cost(train_x,train_y, nn_number_layer, nn_W, nn_b, nn_l2):
    N = np.size(train_x,1)
    x = train_x
    
    for i in range(0,nn_number_layer):
        x= sigm(np.dot(nn_W[i],x) + np.tile(nn_b[i], (1, N)))
        
    total_sum = 0
    for i in range(0,nn_number_layer):
        total_sum = total_sum + sum(sum(nn_W[i]*nn_W[i]))

    cost = sum(sum((x - train_y)**2))/N/2 + nn_l2*total_sum
    return cost

def get_b(nn,sizes,index):
    layer_number = 0
    nn_number_layer= nn[0]
    nn_W= nn[1]
    nn_b= nn[2]
    
    for i in range(0,nn_number_layer):
       if (index < sizes[i+1]) or (index == sizes[i+1]):
            w = nn_W[layer_number][index,:]
            b = nn_b[layer_number][index]
       else:
           index = index - sizes[i+1]
    
    w_b = np.zeros((np.size(w, axis=1)+1,1))
    w_b[0:np.size(w, axis=1)] = w.T
    w_b[-1] = b
    return w_b


def initial_pop(w_b,N):
    d_m = max(w_b.shape)   
    pop_1 = np.zeros((d_m,N))
    w_b1= w_b.reshape((np.size(w_b,axis=0),))  
    pop_1[:,0] = w_b1
    
    for i in range(0,N-1):
        a = (np.random.rand(d_m,1)-0.5)+1
        w_b_new = a*w_b
        w_b_new1= w_b.reshape((np.size(w_b_new,axis=0),))
        pop_1[:,i+1] = w_b_new1
        
    return pop_1

def cal_maturity(x, nn_number_layer, nn_W, nn_b, sizes, alph, m_ratio):
    number_hidden = sum(sizes[1:])
    split_number = 2
     
    b_size = int(x.shape[1]/split_number)
    subtask_m = np.zeros((number_hidden,1))
    sizes[0] = 0
    output=[[[0]],[[0]],[[0]]] 
    
    for bat in range(0,2):
        x_b = x[:,bat*b_size:(bat+1)*b_size]
        for i in range(0,nn_number_layer):
            output[i] = sigm(np.dot(nn_W[i],x_b) + np.tile(nn_b[i], (1, b_size)))
            x_b = output[i]
            subtask_m[sum(sizes[0:i+1]):sum(sizes[0:i+2])] = subtask_m[sum(sizes[0:i+1]):sum(sizes[0:i+2])] + np.expand_dims(g_f(output[i], alph), axis=1)
      
    s_m = np.sort(subtask_m,0)
    s_index = np.argsort(subtask_m,0)
    subtask_index = s_index[:round(number_hidden*m_ratio)]
    return subtask_m, s_m, subtask_index

def put_W(nn,sizes,index,w_b):  
    nn_number_layer= nn[0]
    nn_W= nn[1]
    nn_b= nn[2]
    nn_l2= nn[3]
    layer_number = 0
    
    for i in range(0,nn_number_layer):
       if (index < sizes[i+1]) or (index == sizes[i+1]):
            nn_W[layer_number][index,:] = w_b[0:-1].T
            nn_b[layer_number][index] = w_b[-1]
       else:
           index = index - sizes[i+1]
           
    nn=[]       
    nn.extend((nn_number_layer,nn_W,nn_b, nn_l2))   
    return nn

def get_maturity(x, nn, sizes,index):    
    alph = 0.3
    m_ratio = 0.3    
    nn_number_layer= nn[0]
    nn_W= nn[1]
    nn_b= nn[2]
    
    [m,_,_] = cal_maturity(x, nn_number_layer, nn_W, nn_b, sizes, alph, m_ratio)    
    maturity_value = m[index]
    return maturity_value
    
    
def fun_unsparse(train_x,train_y,nn, sizes,indival,sub_index):
    nn_w = put_W(nn,sizes,sub_index,indival)    
    nn_number_layer= nn_w[0]
    nn_W= nn_w[1]
    nn_b= nn_w[2]
    nn_l2= nn_w[3]
        
    thert = 0.0001
    c = cal_cost(train_x,train_y,nn_number_layer, nn_W, nn_b, nn_l2)
    m = get_maturity(train_x,nn_w, sizes,sub_index)
    f_n = 1/c + thert*m
    return f_n

def DE_unsparse(nn,sizes,train_x,train_y,pop_1,sub_index):
    number_batch = 400
    m = np.size(train_x, axis=1)
    kk = np.random.permutation(range(m))
    x = train_x[:,kk[0:number_batch]]
    y = train_y[:,kk[0:number_batch]]
        
    [w_d,N] = pop_1.shape
    F0 = 0.5
    Np = N
    CR = 0.9    
    
    value = np.zeros((Np,1))
    
    Gm = 50
    Gmin = np.zeros((1,Gm))
    XG_next_1 = np.zeros((w_d,Np))
    XG_next = np.zeros((w_d,Np))
    
    G = 0
    while (G < Gm):
        print(G,Gm,"1")
        for i in range(0,Np):
            a = 1
            b = Np
            dx = np.random.permutation(range(b-a+1)) + a - 1
            j = dx[0]
            k = dx[1]
            p = dx[2]
            if j == i:
                j  = dx[3]
                if k == i:
                     k = dx[3]
                     if p == i:
                           p = dx[3]

            suanzi = np.exp(1-Gm/(Gm + G))
            F = F0*2**suanzi
            son_1 = pop_1[:,p] + F*(pop_1[:,j] - pop_1[:,k])
            XG_next_1[:,i] = son_1
                              
        XG_next_2 = XG_next_1
        TT = (np.random.rand(a,Np)>CR).nonzero()
        XG_next_2[TT] = pop_1[TT]
                 
        for i in range(0,Np):
            a = fun_unsparse(x,y,nn,sizes,XG_next_2[:,i],sub_index)
            b = fun_unsparse(x,y,nn,sizes,pop_1[:,i],sub_index)
            if a > b:
                XG_next[:,i] = XG_next_2[:,i]
                
                value[i] = a
            else:
                XG_next[:,i] = pop_1[:,i]
              
                value[i] = b
    
        value_min, num_min = value.min(0),value.argmin(0)
        Gmin[0,G] = value_min
       
        pop_1 = XG_next
        G = G+1
                
    best_vector_1 = pop_1[:,num_min]        
    return best_vector_1  
  
def saCCDE_all(train_x,train_y,nn,sizes,subtask_index_need):    
    N_P = 20
    nn_number_layer= nn[0]
    nn_W= nn[1]
    nn_b= nn[2]
    nn_l2= nn[3]
    nn_ori = nn
    
    task_number = np.size(subtask_index_need, axis=1)    
    cost_last = cal_cost(train_x,train_y, nn_number_layer, nn_W, nn_b, nn_l2)    
    Cost_DE = np.zeros((task_number,1))
        
    for t in range(0,task_number):
        h = 'the number of subtask is'
        w_b = get_b(nn_ori,sizes,subtask_index_need[t])
        pop_1 = initial_pop(w_b,N_P)
        
        best_vector_1 = DE_unsparse(nn_ori,sizes,train_x,train_y,pop_1,subtask_index_need[t])          
        nn_de= put_W(nn_ori,sizes,subtask_index_need[t],best_vector_1)
        
        nn_de_number_layer= nn_de[0]
        nn_de_W= nn_de[1]
        nn_de_b= nn_de[2]
        nn_de_12= nn_de[3]
    
        Cost_de = cal_cost(train_x,train_y, nn_de_number_layer, nn_de_W, nn_de_b, nn_de_12)
        
        if Cost_de < cost_last:     
            TTTT = 1
            nn = nn_de
            cost_last = Cost_de
            Cost_DE[t,0] = Cost_de
            nn_ori = nn
        else:
            Cost_DE[t,0] = cost_last
            nn_ori = nn
    
    return nn