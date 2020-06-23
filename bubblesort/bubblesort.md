# SORTING NUMBERS WITH BUBBLE SORT USING HANDWRITTEN INPUT

### **Concept:**
- A list of numbers is written by hand and detected using CNN models
- Bubble sort algorithm is applied to sort these numbers
- Bubble sort is a sorting algorithm that works by repeatedly stepping through lists that need to be sorted, comparing each pair of adjacent items and swapping them if they are in the wrong order. 
- This passing procedure is repeated until no swaps are required, indicating that the list is sorted.
- Bubble sort gets its name because smaller elements bubble toward the top of the list.

### **Required Packages:**
```
pip install opencv-python==3.4.1
pip install tensorflow==1.13.1
pip install Keras==2.0.6
pip install python==3.6.5
```
### **Model:**
- Model used for this code can be downloaded from the following link: <br>
    https://gitlab.com/school-of-curious/reshma-ramesh-babu/blob/develope/bubblesort/model_mnist3.h5
- Save it in the same folder as the python file and run the code

### **Approach:**
- Bubble sort is a sorting algorithm that works by repeatedly stepping through lists that need to be sorted, comparing each pair of adjacent items and swapping them if they are in the wrong order. 
- This passing procedure is repeated until no swaps are required, indicating that the list is sorted.
- Bubble sort gets its name because smaller elements bubble toward the top of the list.

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

a = 100
b = 100
#definition for bubble sort function
def bubbleSort(output,arr,count1):
    global a,b
    swapFlag = True
    cv2.putText(output,"Comparing adjacent elements,",(100,75),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(output,"Swapping if first element less than second",(100,125),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2,cv2.LINE_AA)
    a = 100
    b = 200
    #displaying original array
    cv2.putText(output,"Array:",(a,b),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
    for k in range(len(arr)):
        if k==0:
            a+=150
            cv2.putText(output,str(arr[k]),(a,b),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
        else:
            a+=100
            cv2.putText(output,str(arr[k]),(a,b),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
    j = 1
    while swapFlag:
        swapFlag= False
        for i in range(len(arr)-1):
            if count1 >= i:
                #comparing adjacent elements and swapping if first less than second on key press
                if arr[i] > arr[i+1]:
                    arr[i], arr[i+1] = arr[i+1], arr[i]
                    a = 100
                    swapFlag = True
                    a = 100
                    cv2.putText(output,"Swapping "+str(arr[i])+" and "+str(arr[i+1]),(a,b + 50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2,cv2.LINE_AA)
                    a = 100
                    b = b + 100
                    for k in range(len(arr)):
                        if k==0:
                            a+=120
                            cv2.putText(output,str(arr[k]),(a,b),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
                        else:
                            a+=80
                            cv2.putText(output,str(arr[k]),(a,b),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
                '''output[np.where((output==[0,255,255]).all(axis=2))]=[0,0,0]
                output[np.where((output==[255,0,255]).all(axis=2))]=[0,0,0]'''
                                
    a = 50
    b = b + 100
    cv2.putText(output,"Sorted array:",(a,b),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
    for k in range(len(arr)):
        if k==0:
            a+=200
            cv2.putText(output,str(arr[k]),(a,b),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
        else:
            a+=100
            cv2.putText(output,str(arr[k]),(a,b),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
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
    #for showing code generation
    if cv2.waitKey(1) & 0xFF == ord('b'):
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
        #displaying every step of bubble sort on pressing the key a
        if cv2.waitKey(1) & 0xFF == ord('a'):
            count1 = count1 + 1
            bubbleSort(output,arr,count1)
        ARR = arr
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
        #to show basic code for bubble sort in a new frame
        if count >= 1:
            frame5=np.zeros((1000,900,3),np.uint8)
            cv2.putText(frame5,"def bubbleSort(arr):",(100,60),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    swapFlag = True",(100,100),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    print(arr)",(100,140),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    j = 1",(100,180),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    while swapFlag:",(100,220),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        swapFlag= False",(100,260),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        for i in range(len(arr)-1):",(100,300),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            c = 0",(100,340),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            if swapFlag == False:",(100,380),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"                c += 1",(100,420),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            if c < 2:",(100,460),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"                print('step',j)",(100,500),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            if arr[i] > arr[i+1]:",(100,540),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"                arr[i], arr[i+1] = arr[i+1], arr[i]",(100,580),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"                print('Swapped:',arr[i],'with',arr[i+1])",(100,620),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"                swapFlag = True",(100,640),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"                print(arr)",(100,680),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        j = j + 1",(100,720),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    print('Sorted array:',arr)",(100,760),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"arr = "+str(arr),(100,800),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"bubbleSort(arr)",(100,840),0,0.8,(10,100,255),2)
            cv2.imshow("Bubble sort",frame5)
    except:
        pass
    cv2.imshow('frame1', img)
    cv2.imshow('frame', frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break        
cap.release()
#code for code generation for bubble sort
f1 = open("bubble_sort.py","w+")
code='''
def bubbleSort(arr):
    swapFlag = True
    print(arr)
    j = 1
    while swapFlag:
        swapFlag= False        
        for i in range(len(arr)-1):
            c = 0
            if swapFlag == False:
                c += 1
            if c < 2:
                print("step",j)
            if arr[i] > arr[i+1]:
                arr[i], arr[i+1] = arr[i+1], arr[i]

                print("Swapped: {} with {}".format(arr[i], arr[i+1]))
                swapFlag = True
                print(arr)
        j = j + 1
    print("Sorted array:",arr)
arr = '''+str(ARR)+'''
bubbleSort(arr)'''
f1.write(code)
f1.close()
```
- Run the following code
```
python3 bubblesorthw.py
```
### **Operation:**
- Bubble sort compares every adjacent element and swaps them if they are in the wrong order
- On running the above code, on pressing a, the elements are sorted in bubble sort and displayed step by step
- On pressing b, the basic code for bubble sort is generated and displayed in a new frame

### **Demo video link:**

https://www.dropbox.com/s/tbfxxzx6fmgjp1e/bubsort.mp4?dl=0