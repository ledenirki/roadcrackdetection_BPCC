#Başarı değerlendirme yöntemleri

def succes_rate(y_pred,y_test):   
    a, b= y_test.shape
    tn,tp,fn,fp= 0,0,0,0
    for i in range(a):
        for j in range(b):
            yp=y_pred[i,j]
            y=y_test[i,j]  
            
            #Tahmin edilen değerin ve gerçek sınıf değerinin 1 olması yani True-Possitive durumu
            if yp==1 and y==1:      
                tp=tp+1;
            
            #Tahmin edilen değerin ve gerçek sınıf değerinin 0 olması yani True-Negative durumu
            elif yp==0 and y==0:    
                tn=tn+1;
                
            #Tahmin edilen değerin 0 ve gerçek sınıf değerinin 1 olması yani False-Negative durumu
            elif yp==0 and y==1:    
                fn=fn+1;
                
            #Tahmin edilen değerin 1 ve gerçek sınıf değerinin 0 olması yani False-Possitive durumu
            elif yp==1 and y==0:    
                fp=fp+1;  
    
    #True-Possitive ve False-Possitive değerinin 0 olması durumunda başarı yüzdesi belirlenmeyeceği için başarı yüzdesi doğrudan 0'dır
    if tp==0 and fp==0:         
        f1=0
    
    #True-Possitive ve False-Negative değerinin 0 olması durumunda başarı yüzdesi belirlenmeyeceği için başarı yüzdesi doğrudan 0'dır
    elif tp==0 and fn==0:       
        f1=0
    else:
        precision= tp/(tp+fp)   #Kesinlik hesabı
        recall= tp/(tp+fn)      #Duyarlılık hesabı
        
        #Kesinlik ve Duyarlılık değerlerinin 0 olması durumunda başarı yüzdesi belirlenemeyeceği için başarı yüzdesi doğrudan 0'dır
        if precision==0 and recall==0: 
            f1=0
        else:
            #Başarı yüzdesi yani kesinlik ve doğruluk değerlerinin harmonik ortalaması
            f1= 2*(precision*recall)/(precision+recall)*100 
        
    print('Doğruluk olasılığı: %{} '.format(f1))
    print('Precision: %{} '.format(precision*100))
    print('Recall: %{} '.format(recall*100))
    return f1  