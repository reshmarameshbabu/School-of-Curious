# SORTING NUMBERS WITH MERGESORT USING HANDWRITTEN INPUT

### **Concept:**
- A list of numbers is written by hand and detected using CNN models
- Merge sort algorithm is applied to sort these numbers
- Merge sort keeps on dividing the list into equal halves until it can no more be divided. If there is only one element in the list, it is sorted. Then, merge sort combines the smaller sorted lists keeping the new list sorted too.

### **Required Packages:**
```
pip install opencv-python==3.4.1
pip install tensorflow==1.13.1
pip install Keras==2.0.6
pip install python==3.6.5
```
### **Model:**
- Model used for this code can be downloaded from the following link: <br>
    https://gitlab.com/school-of-curious/reshma-ramesh-babu/blob/develope/mergesort/model_mnist3.h5
- Save it in the same folder as the python file and run the code

### **Approach:**
- Merge sort is a divide-and-conquer algorithm based on the idea of breaking down a list into several sub-lists until each sublist consists of a single element and merging those sublists in a manner that results into a sorted list. 
- Basic idea:
    * Divide the unsorted list into n sublists, each containing 1 element.
    * Take adjacent pairs of two singleton lists and merge them to form a list of 2 elements.
    * N will now convert into n/2 lists of size 2.
    * Repeat the process till a single sorted list of obtained. 
- Merging:
    * While comparing two sublists for merging, the first element of both lists is taken into consideration. 
    * While sorting in ascending order, the element that is of a lesser value becomes a new element of the sorted list.
    *  This procedure is repeated until both the smaller sublists are empty and the new combined sublist comprises all the elements of both the sublists.

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

#definition of merge sort function
def merge_sort(alist, start, end,x,y,offsetx,offsety,level,flag):
    #Sorts the list from indexes start to end - 1 inclusive.
    if end - start > 1:
        mid = (start + end)//2
        posx = x
        cv2.arrowedLine(output,(x + 100,y - 90),(x - offsetx + 30,y-30),(255,0,0),2)
        cv2.arrowedLine(output,(x + 100,y - 90),(x + offsetx + 50,y-30),(255,0,0),2)
        # flag is 0 for left sub half and 1 for right sub half, to display them in different colours
        if flag == 0:
            for i in range(start,mid):
                cv2.putText(output,str(alist[i]),(posx - offsetx, y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
                posx+=50
            for i in range(mid,end):
                cv2.putText(output,str(alist[i]),(posx + offsetx,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
                posx+=50
        else:
            for i in range(start,mid):
                cv2.putText(output,str(alist[i]),(posx - offsetx, y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)
                posx+=50
            for i in range(mid,end):
                cv2.putText(output,str(alist[i]),(posx + offsetx,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)
                posx+=50
        
        #every iteration of the merge sort is displayed on key press
        if count != 0 and count <= level:
            merge_sort(alist, start, mid,x - offsetx,y + 100,offsetx - 70,offsety-50,level+1,0)
            merge_sort(alist, mid, end,x + offsetx,y + 100,offsetx - 70,offsety-50,level+1,1)
            merge_list(alist, start, mid, end,x,offsetx + offsety + 50,offsetx,flag)
            cv2.arrowedLine(output,(x - offsetx + 20,y+50),(x+20,offsetx+offsety+10),(255,0,0),2)
            cv2.arrowedLine(output,(x + offsetx + 20,y+50),(x+20,offsetx+offsety+10),(255,0,0),2)

#function to merge the splitted sub lists
def merge_list(alist, start, mid, end,x,y,offsetx,flag):
    left = alist[start:mid]
    right = alist[mid:end]
    
    k = start
    i = 0
    j = 0
    #to compare left and right sub lists and merge
    while (start + i < mid and mid + j < end):
        if (left[i] <= right[j]):
            alist[k] = left[i]
            i = i + 1
        else:
            alist[k] = right[j]
            j = j + 1
        k = k + 1
    #merging the remaining elements in both left and right sub lists
    if start + i < mid:
        while k < end:
            alist[k] = left[i]
            i = i + 1
            k = k + 1
    else:
        while k < end:
            alist[k] = right[j]
            j = j + 1
            k = k + 1 

    posx = x
    if flag ==0:
        for i in range(start,end):
            cv2.putText(output,str(alist[i]),(posx,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
            posx+=50
    else:
        for i in range(start,end):
            cv2.putText(output,str(alist[i]),(posx,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)
            posx+=50

global ARR

import operator
cap = cv2.VideoCapture(0)
count = 0
count1 = 0
while(True):
    #reading frames from camera
    ret, frame1 = cap.read()
    frame = frame1.copy()
    frame_new= frame1.copy()
    #key press m for every itereation of mergesort
    if cv2.waitKey(1) & 0xFF == ord('m'):
            count += 1
    #key press s to display code generation
    if cv2.waitKey(1) & 0xFF == ord('s'):
            count1 += 1
    th1 = 78
    ret, img = cv2.threshold(frame, th1, 255, cv2.THRESH_BINARY_INV)
    cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imagenew,contours, hierarchy = cv2.findContours(cvt,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros((1000,1000,3),np.uint8)
    thisdict = {}
    
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
        #obtaining read string as a list
        arr=list(s)
        #splitting commas from the given list
        arr = s.split(',')
        ARR = arr
        cv2.putText(output,"Array",(100,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
        posx = 300
        y = 50
        for i in range(0,len(arr)):
            cv2.putText(output,str(arr[i]),(posx,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
            posx+=50
        
        merge_sort(arr, 0, len(arr),400,150,150,500,1,0)
        #sorted array
        cv2.putText(output,"Sorted array:",(100,850),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
        posx = 300
        y = 850
        for i in range(0,len(arr)):
            cv2.putText(output,str(arr[i]),(posx,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
            posx+=50
        #displaying basic cocde for merge sort in a new frame
        if count1 >= 1:
            frame5=np.zeros((1000,4000,3),np.uint8)
            cv2.putText(frame5,"def merge_sort(alist, start, end):",(100,60),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    if end - start > 1:",(100,100),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        mid = (start + end)//2",(100,140),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        merge_sort(alist, start, mid)",(100,180),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        merge_sort(alist, mid, end)",(100,220),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        merge_list(alist, start, mid, end)",(100,260),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"def merge_list(alist, start, mid, end):",(100,300),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    left = alist[start:mid]",(100,340),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    right = alist[mid:end]",(100,380),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    k = start",(100,420),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    i = 0",(100,460),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    j = 0",(100,500),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    while (start + i < mid and mid + j < end):",(100,540),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        if (left[i] <= right[j]):",(100,580),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            alist[k] = left[i]",(100,620),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            i = i + 1",(100,640),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        else:",(100,680),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            alist[k] = right[j]",(100,720),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            j = j + 1",(100,760),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        k = k + 1",(100,800),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    if start + i < mid:",(100,840),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        while k < end:",(100,880),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            alist[k] = left[i]",(100,920),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            i = i + 1",(100,960),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            k = k + 1",(100,1000),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    else:",(100,1400),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        while k < end:",(100,1800),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            alist[k] = right[j]",(100,2200),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            j = j + 1",(100,2600),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            k = k + 1",(100,3000),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"alist ="+str(arr),(100,3400),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"merge_sort(alist, 0, len(alist))",(100,3800),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"print(alist)",(100,4200),0,0.8,(10,100,255),2)
            cv2.imshow("Merge sort",frame5)

    except:
        pass
    cv2.imshow('frame1', img)
    cv2.imshow('frame', frame1)
    cv2.imshow('output',output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break        
cap.release()
#code for code generation of merge sort
f1 = open("merge_sort.py","w+")

code = '''
def merge_sort(alist, start, end):
    if end - start > 1:
        mid = (start + end)//2
        merge_sort(alist, start, mid)
        merge_sort(alist, mid, end)
        merge_list(alist, start, mid, end)
 
def merge_list(alist, start, mid, end):
    left = alist[start:mid]
    right = alist[mid:end]
    k = start
    i = 0
    j = 0
    while (start + i < mid and mid + j < end):
        if (left[i] <= right[j]):
            alist[k] = left[i]
            i = i + 1
        else:
            alist[k] = right[j]
            j = j + 1
        k = k + 1
    if start + i < mid:
        while k < end:
            alist[k] = left[i]
            i = i + 1
            k = k + 1
    else:
        while k < end:
            alist[k] = right[j]
            j = j + 1
            k = k + 1
 
 
alist = '''+str(ARR)+'''
merge_sort(alist, 0, len(alist))
print('Sorted list: ', end='')
print(alist)'''

f1.write(code)
f1.close()
```
- Run the following code
```
python3 mergesorthw.py
```
### **Operation:**
- Merge sort splits the given array into halves and merges them while sorting them
- On running the above code, the frame where the numbers are recognised is displayed, on pressing m repeatedly in the output frame, the list is splitted and merged while sorting simultaneously
- On pressing s, the basic code for merge sort is generated and displayed in a new frame

### **Demo video link:**

https://www.dropbox.com/s/v4mgh3zpo0y4mhl/mergesort.mkv?dl=0

