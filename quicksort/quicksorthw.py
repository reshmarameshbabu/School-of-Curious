#importing libraries and dependencies
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense

#creating the model
def create_model():
    model = Sequential()
    model.add(Convolution2D(16, 5, 5, activation='relu', input_shape=(28,28, 3)))
    model.add(MaxPooling2D(2, 2))

    model.add(Convolution2D(32, 5, 5, activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))

    model.add(Dense(11, activation='softmax'))
    return model

#loading model
model = create_model()
model.load_weights('/home/reshma/admatic/quicksort/model_mnist3.h5')

x = 400
y = 200
a = 100
b = 100
def partition(arr,low,high):
    global x,y,a,b
    font=cv2.FONT_HERSHEY_SIMPLEX
    i = ( low-1 )         
    pivot = arr[high]
    if b < 500:
        b += 20
    for j in range(low , high): 
        x = 100
        cv2.putText(output,"pivot="+str(pivot),(x,y),font,0.65,(0,255,255),2)
        x = 400
        if arr[j] <= pivot: 
            for k in range(len(arr)):
                if i==0:
                    x+=80
                    cv2.putText(output,str(arr[k]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
                else:
                    x+=20
                    cv2.putText(output,str(arr[k]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
            i = i+1 
            x = 200
            y = y + 100
            cv2.putText(output,"Swapping "+str(arr[i])+" and "+str(arr[j]),(x,y),font,0.65,(0,255,255),2)
            arr[i],arr[j] = arr[j],arr[i] 
    x = 200
    y = y + 100
    cv2.putText(output,"Swapping "+str(arr[i+1])+" and "+str(arr[high]),(x,y),font,0.65,(0,255,255),2)
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    x = 400
    y = y + 50
    for k in range(len(arr)):
        if i==0:
            x+=80
            cv2.putText(output,str(arr[k]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
        else:
            x+=20
            cv2.putText(output,str(arr[k]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
    return ( i+1 ) 
c = 1
def quickSort(arr,low,high): 
    if low < high: 
        pi = partition(arr,low,high)
        quickSort(arr, low, pi-1) 
        quickSort(arr, pi+1, high) 
global ARR
import operator
cap = cv2.VideoCapture(0)
st = ""
count = 0
while(True):
    #reading frames from camera
    ret, frame1 = cap.read()
    
    frame = frame1.copy()
    frame_new= frame1.copy()
    
    th1 = 78
    ret, img = cv2.threshold(frame, th1, 255, cv2.THRESH_BINARY_INV)
    cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imagenew,contours, hierarchy = cv2.findContours(cvt,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros((1000,1000,3),np.uint8)

    thisdict = {}
    if cv2.waitKey(1) & 0xFF == ord('s'):
        count += 1
    
    flag=0
    noted_y=0
    mylist = []
    for c in contours:
        (x, y, w, h)= cv2.boundingRect(c)
        if (w>20) or (h>20):
            mylist.append((x,y,w,h))
    
    th2 = 82 #change thresholding of every contour
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
        ret, gray = cv2.threshold(img1,th2,255,cv2.THRESH_BINARY )
        try:
            im = cv2.resize(gray, (28,28))
            #cv2.imshow('img',gray)

            ar = np.array(im).reshape((28,28,3))
            ar = np.expand_dims(ar, axis=0)
            prediction = model.predict(ar)[0]
            #prediction of class labels
            for i in range(0,12):
                if prediction[i]==1.0:
                    if i==0:
                        j= ","
                    if i==1:
                        j= "0"
                    if i==2:
                        j= "1"
                    if i==3:
                        j= "2"
                    if i==4:
                        j= "3"
                    if i==5:
                        j= "4"
                    if i==6:
                        j= "5"
                    if i==7:
                        j= "6"
                    if i==8:
                        j= "7"
                    if i==9:
                        j= "8"
                    if i==10:
                        j= "9"
                     
            #printing prediction        
                    cv2.putText(frame1, j, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
                    thisdict[x]= str(j)
        except:
            d=0

    sort = sorted(thisdict.items(), key=operator.itemgetter(0))
    s = ""
    for x in range(0,len(sort)):
        s=s+str(sort[x][1])
        cv2.putText(frame1, s, (100,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)  
    try:
        arr=list(s)
        n=len(arr)
        for i in range(n):
            if arr[i] == ',':
                n = n - 1
        arr = s.split(',')
        if cv2.waitKey(1) & 0xFF == ord('a'):
                quickSort(arr,0,n-1)
        ARR=arr
        s="".join(arr)
        x=100
        cv2.putText(frame1,"Sorted array:",(x,350),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
        for i in range(0,len(arr)):
            if i==0:
                x+=100
                cv2.putText(frame1,str(arr[i]),(x,400),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
            else:
                x+=50
                cv2.putText(frame1,str(arr[i]),(x,400),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
            #cv2.putText(frame1,str(arr[i]),(x,400),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
        if count >= 1:
            frame5=np.zeros((1500,1200,3),np.uint8)
            cv2.putText(frame5,"def partition(arr,low,high):",(50,60),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    i = ( low-1 )",(50,100),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    pivot = arr[high]",(50,140),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    print('Pivot:',pivot)",(50,180),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    for j in range(low , high):",(50,220),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        if arr[j] <= pivot:",(50,260),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            print(arr)",(50,300),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            i = i+1 ",(50,340),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            print('Swapping',arr[i],'and',arr[j])",(50,380),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            arr[i],arr[j] = arr[j],arr[i]",(50,420),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    print('Swapping',arr[i+1],'and',arr[high])",(50,460),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    arr[i+1],arr[high] = arr[high],arr[i+1]",(50,500),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    print(arr)",(50,540),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    return ( i+1 )",(50,580),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"def quickSort(arr,low,high):",(50,620),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    if low < high:",(50,680),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        pi = partition(arr,low,high) ",(50,720),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        quickSort(arr, low, pi-1)",(50,760),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        quickSort(arr, pi+1, high)",(50,800),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"arr ="+str(arr),(50,840),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"n = len(arr)",(50,880),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"quickSort(arr,0,n-1)",(50,920),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"print ('Sorted array is:'')",(50,960),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"print(arr)",(50,1000),0,0.8,(10,100,255),2)
            cv2.imshow("Quick sort",frame5)       
    except:
        pass
    cv2.imshow('frame1', img)
    #cv2.imshow('fram', im)
    cv2.imshow('frame', frame1)
    cv2.imshow('output',output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break        
cap.release()

f1 = open("quick_sort.py","w+")
code='''
def partition(arr,low,high): 
    i = ( low-1 )         
    pivot = arr[high]     
    print("Pivot:",pivot)
    for j in range(low , high): 
   
        if arr[j] <= pivot: 
            print(arr)
            i = i+1 
            print("Swapping",arr[i],"and",arr[j])
            arr[i],arr[j] = arr[j],arr[i] 
    print("Swapping",arr[i+1],"and",arr[high])   
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    print(arr)
    return ( i+1 ) 
  
def quickSort(arr,low,high): 
    if low < high:  
        pi = partition(arr,low,high) 
        
        quickSort(arr, low, pi-1) 
        quickSort(arr, pi+1, high) 
  
# Driver code to test above 
arr = '''+str(ARR)+'''
n = len(arr) 
quickSort(arr,0,n-1) 
print ("Sorted array is:") 
print(arr)'''
f1.write(code)
f1.close()