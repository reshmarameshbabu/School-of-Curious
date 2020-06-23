import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense

def create_model():
    model = Sequential()
    model.add(Convolution2D(16, 5, 5, activation='relu', input_shape=(28,28, 3)))
    model.add(MaxPooling2D(2, 2))

    model.add(Convolution2D(32, 5, 5, activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))

    model.add(Dense(5, activation='softmax'))
    return model

def value(r): 
    if (r == 'I'): 
        return 1
    if (r == 'V'):  
        return 5
    if (r == 'X'): 
        return 10
    if (r == 'L'): 
        return 50
    if (r == 'C'): 
        return 100
    if (r == 'D'): 
        return 500
    if (r == 'M'): 
        return 1000
    return -1

def romanToDecimal(str): 
    res = 0
    i = 0
  
    while (i < len(str)): 
  
        # Getting value of symbol s[i] 
        s1 = value(str[i]) 
  
        if (i+1 < len(str)): 
  
            # Getting value of symbol s[i+1] 
            s2 = value(str[i+1]) 
  
            # Comparing both values 
            if (s1 >= s2): 
  
                # Value of current symbol is greater 
                # or equal to the next symbol 
                res = res + s1 
                i = i + 1
            else: 
  
                # Value of current symbol is greater 
                # or equal to the next symbol 
                res = res + s2 - s1 
                i = i + 2
        else: 
            res = res + s1 
            i = i + 1
  
    return res 

model = create_model()
model.load_weights('/home/reshma/admatic/romannumber/model_mnist6.h5')

import operator
cap = cv2.VideoCapture(1)

while(True):
    ret, frame1 = cap.read()
    frame = frame1.copy()
    frame_new= frame1.copy()
    
    ret, img = cv2.threshold(frame, 78, 255, cv2.THRESH_BINARY_INV)
    cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imagenew,contours, hierarchy = cv2.findContours(cvt,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    mylist = []
    thisdict = {}
    
    for c in contours:
        (x, y, w, h)= cv2.boundingRect(c)
        if (w>45) or (h>45):
            mylist.append((x,y,w,h))
            
    for i in range(0, len(mylist)):
        x = mylist[i][0]
        y = mylist[i][1]
        w = mylist[i][2]
        h = mylist[i][3]
        if h/w>3:
            x=x-10
            w=w+20
        if w/h>3:
            y=y-60
            h=h+110
        y=y-27
        x=x-25
        w=w+50
        h=h+54
        cv2.rectangle(frame1,(x,y),(x+w,y+h), (0,0, 255), 2)
        img1 = frame_new[y:y+h, x:x+w]
        ret, gray = cv2.threshold(img1,108,255,cv2.THRESH_BINARY )
        try:
            #prediction of class labels
            im = cv2.resize(gray, (28,28))
            ar = np.array(im).reshape((28,28,3))
            ar = np.expand_dims(ar, axis=0)
            prediction = model.predict(ar)[0]

            #prediction of class labels
            for i in range(0,6):
                if prediction[i]==1.0:
                    if i==0:
                        j= "C"
                    if i==1:
                        j= "I"
                    if i==2:
                        j= "L"
                    if i==3:
                        j= "V"
                    if i==4:
                        j= "X"
            #printing prediction
                    cv2.putText(frame1, j, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
                    thisdict[x]= str(j)
        except:
            d=0

    sort = sorted(thisdict.items(), key=operator.itemgetter(0))
    s = ""
    for x in range(0,len(sort)):
        s=s+str(sort[x][1])
        #cv2.putText(frame1, s, (100,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)  

    font = cv2.FONT_HERSHEY_SIMPLEX

    ans=romanToDecimal(str(s))
    #print(fullexp)
    cv2.putText(frame1,'in roman '+str(s) +' ==> '+str(ans), (100,100), font, 1, (0,0,255), 2, cv2.LINE_AA)
        
    
    cv2.imshow('frame', frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break        
cap.release()
