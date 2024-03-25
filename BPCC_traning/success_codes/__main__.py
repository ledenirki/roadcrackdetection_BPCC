#Success evaluation methods

def succes_rate(y_pred,y_test):   
    a, b= y_test.shape
    tn,tp,fn,fp= 0,0,0,0
    for i in range(a):
        for j in range(b):
            yp=y_pred[i,j]
            y=y_test[i,j]  
            
            if yp==1 and y==1:      
                tp=tp+1;
            
            elif yp==0 and y==0:    
                tn=tn+1;
                
            elif yp==0 and y==1:    
                fn=fn+1;
                
            elif yp==1 and y==0:    
                fp=fp+1;  
    
    if tp==0 and fp==0:         
        f1=0
    
    elif tp==0 and fn==0:       
        f1=0
    else:
        precision= tp/(tp+fp)   #Kesinlik hesabı
        recall= tp/(tp+fn)      #Duyarlılık hesabı
        
        if precision==0 and recall==0: 
            f1=0
        else:
            f1= 2*(precision*recall)/(precision+recall)*100 
        
    print('F1-Score: %{} '.format(f1))
    print('Precision: %{} '.format(precision*100))
    print('Recall: %{} '.format(recall*100))
    return f1  
