# SORTING NUMBERS WITH INSERTION SORT USING HANDWRITTEN INPUT

### **Concept:**
- A list of numbers is written by hand and detected using CNN models
- Insertion sort compares the current element with the largest value in the sorted array. If the current element is greater, then it leaves the element in its place and moves on to the next element else it finds its correct position in the sorted array and moves it to that position.
- This is done by shifting all the elements, which are larger than the current element, in the sorted array to one position ahead.

### **Required Packages:**
```
pip install opencv-python==3.4.1
pip install tensorflow==1.13.1
pip install Keras==2.0.6
pip install python==3.6.5
```

### **Model:**
- Model used for this code can be downloaded from the following link: <br>
    https://gitlab.com/school-of-curious/reshma-ramesh-babu/blob/develope/insertionsort/model_mnist3.h5
- Save it in the same folder as the python file and run the code

### **Approach:**
#### Insertion sort working:

1. The first step involves the comparison of the element in question with its adjacent element.
2. And if at every comparison reveals that the element in question can be inserted at a particular position, then space is created for it by shifting the other elements one position to the right and inserting the element at the suitable position.
3. The above procedure is repeated until all the element in the array is at their apt position.


### **Code:**
- Create a new python file with the following code
```python
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
model.load_weights('/home/reshma/admatic/model_mnist3.h5')
x = 100
y = 250
p = 300
q = 275
#definition of insertion sort function
def insertionSort(output,arr,count1):
    global x , y , p, q
    x = 100
    y = 100
    cv2.putText(output,"Move elements greater than key(in front of key) ",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(output,"to one position ahead of current position and",(x,y+30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(output,"insert key at that position",(x,y+60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2,cv2.LINE_AA)
    x = 100
    y = 250
    #printing original array
    cv2.putText(output,"Array:",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
    for k in range(len(arr)):
        if k==0:
            x+=150
            cv2.putText(output,str(arr[k]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
        else:
            x+=100
            cv2.putText(output,str(arr[k]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
    # Traverse through 1 to len(arr) 
    y = 50
    for i in range(1, len(arr)): 
        if count1 >= i:
            output[np.where((output==[0,255,255]).all(axis=2))]=[0,0,0]
            output[np.where((output==[255,0,255]).all(axis=2))]=[0,0,0]
            key = arr[i] 
            x = 100
            y = y + 50
            #printing key element chosen at every step
            cv2.putText(output,"Key:"+str(key),(x - 10,y+60*(i+1)+80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2,cv2.LINE_AA)
            cv2.putText(output,"Move elements greater than "+str(key)+" and before it, one step right",(x + 80,y+60*(i+1)+90),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2,cv2.LINE_AA)            
            cv2.putText(output,"Insert "+str(key)+" at that position",(x + 80,y+60*(i+1)+130),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2,cv2.LINE_AA)            
            # Move elements of arr[0..i-1], that are 
            # greater than key, to one position ahead 
            # of their current position 
            j = i-1
            while j >= 0 and key < arr[j] : 
                arr[j+1] = arr[j] 
                j -= 1
            arr[j+1] = key 
            x = 100
            
            for k in range(len(arr)):
                if k == 0:
                    x+=120
                    cv2.putText(output,str(arr[k]),(x,y+60*(i+1)+170),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
                else:
                    x+=80
                    cv2.putText(output,str(arr[k]),(x,y+60*(i+1)+170),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
    x = 100
    y = 800
    cv2.putText(output,"Sorted array:",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
    for k in range(len(arr)):
        if k==0:
            x+=200
            cv2.putText(output,str(arr[k]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
        else:
            x+=100
            cv2.putText(output,str(arr[k]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
    cv2.imshow('output',output)
global ARR  
import operator
cap = cv2.VideoCapture(1)
st = ""
count = 0
count1 = 0
while(True):
    #reading frames from camera
    ret, frame1 = cap.read()
    output = np.zeros((1000,1000,3),np.uint8)
    frame = frame1.copy()
    frame_new= frame1.copy()    
    
    th1 = 78
    ret, img = cv2.threshold(frame, th1, 255, cv2.THRESH_BINARY_INV)
    cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imagenew,contours, hierarchy = cv2.findContours(cvt,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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
        #obtaining read string as list
        arr=list(s)
        #splitting commas from list
        arr = s.split(',')
        #every step of insertion sort is printed on pressing key a
        if cv2.waitKey(1) & 0xFF == ord('a'):
            count1 = count1 + 1
            insertionSort(output,arr,count1)
        ARR=arr
        s="".join(arr)
        x=100
        #printing sorted array in frame
        cv2.putText(frame1,"Sorted array:",(x,350),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
        for i in range(0,len(arr)):
            if i==0:
                x+=100
                cv2.putText(frame1,str(arr[i]),(x,400),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
            else:
                x+=50
                cv2.putText(frame1,str(arr[i]),(x,400),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
        #printing sorted array in output frame
        cv2.putText(output,"Sorted array:",(x,350),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
        for i in range(0,len(arr)):
            if i==0:
                x+=100
                cv2.putText(output,str(arr[i]),(x,400),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
            else:
                x+=50
                cv2.putText(output,str(arr[i]),(x,400),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA) 
    except:
        pass
    #to print basic insertion sort code in a new frame
    if count >= 1:
            frame5=np.zeros((1500,900,3),np.uint8)
            cv2.putText(frame5,"def insertionSort(arr):",(50,60),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    print('Move elements greater than key (before key)",(50,100),0,0.8,(10,100,255),2)
            cv2.putText(frame5," to one position ahead of current position and insert key at that position')",(50,140),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    for i in range(1, len(arr)): ",(50,180),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        key = arr[i] ",(50,220),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        print('Key=',key)",(50,260),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        j = i-1",(50,300),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        while j >=0 and key < arr[j] : ",(50,340),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"                arr[j+1] = arr[j] ",(50,380),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"                j -= 1",(50,420),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        arr[j+1] = key",(50,460),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        print(arr)",(50,500),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"arr ="+str(arr),(50,540),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"insertionSort(arr) ",(50,580),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"print ('Sorted array is:') ",(50,620),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"print(arr)",(50,680),0,0.8,(10,100,255),2)
            cv2.imshow("Insertion sort",frame5)   
    
    cv2.imshow('frame1', img)
    #cv2.imshow('fram', im)
    cv2.imshow('frame', frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break        
cap.release()
#code for basic code generation of insertion sort
f1 = open("insertion_sort.py","w+")
code='''
def insertionSort(arr): 
    print("Move elements greater than key (before key) to one position ahead of current position and insert key at that position")
    for i in range(1, len(arr)): 
  
        key = arr[i] 
        print("Key=",key)
        
        j = i-1
        while j >=0 and key < arr[j] : 
                arr[j+1] = arr[j] 
                j -= 1
        arr[j+1] = key 
        print(arr)
  
arr = '''+str(ARR)+'''
insertionSort(arr) 
print ("Sorted array is:") 
print(arr) '''
f1.write(code)
f1.close()

```
- Run the following code
```
python3 insertionsorthw.py
```
### **Operation:**
- Insertion sort works on the logic that, all elements greater than the current element are shifted by one position ahead.
- On running the above code, on pressing a, the elements are sorted in insertion sort and displayed step by step
- On pressing s, the basic code for insertion sort is generated and displayed in a new frame

### **Demo video link:**

https://www.dropbox.com/s/1w0sgis898zp2rw/inssort.mp4?dl=0