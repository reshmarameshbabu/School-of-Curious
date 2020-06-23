# SUBTRACTION OF LARGE NUMBERS USING HANDWRITTEN INPUT

### **Concept:**
- Two numbers are written and the numbers are detected using CNN models
- These two numbers are subtracted using algorithm
- Borrow and result are displayed step by step accordingly

### **Required Packages:**
```
pip install opencv-python==3.4.1
pip install tensorflow==1.13.1
pip install Keras==2.0.6
pip install python==3.6.5
```

### **Model:**
- Model used for this code can be downloaded from the following link: 
    https://gitlab.com/school-of-curious/reshma-ramesh-babu/blob/develope/subtraction/model_mnist4.h5
- Save it in the same folder as the python file and run the code

### **Approach:**
- Write down the larger number first and the smaller number directly below it, making sure to line up the columns and do subtractions one column at a time.
- If a column has a smaller number on top, subtraction is done by borrowing: 1 is subtracted from the next number and 10 is added to the current number
- If the next number is 0 and one cannot be subtracted, we make the 0 to a 10 by borrowing it first, then continue as usual. 

### CODE:
- Create a new python file with the following code
```python
import cv2 
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense
cline = 0
aline = 0
#function to return handwritten inputs as strings
def print_result(x_p,y_p,val):
    diff = 0
    for i in range(len(y_p)):
        if(abs(y_p[0]-y_p[i])>diff):
            diff = abs(y_p[0]-y_p[i])
    err = diff - int(diff*0.2)
    min_y = min(y_p)
    s1= ''
    s2 = ''
    for i in range(len(val)):
        if((y_p[i]-min_y)>err):
            s2+=val[i]
        else:
            s1+=val[i]
    
    return s1,s2
#function to append values at appropriate positions
def insert_val(x_p,y_p,val,v,x,y):
    if(len(x_p)==0):
        x_p.append(x)
        y_p.append(y)
        val.append(v)
    else:
        x_p.append(x)
        y_p.append(y)
        val.append(v)
        i=len(x_p)-1
        while i>0:
           if x_p[i]<x_p[i-1]:
               x_p[i],x_p[i-1] =x_p[i-1],x_p[i]
               y_p[i],y_p[i-1] =y_p[i-1],y_p[i]
           else:
               break
           i-=1
        j=len(val)-1
        while j>i:
            val[j] = val[j-1]
            j-=1
        val[j]=v
    return x_p,y_p,val
#creating model
def create_model():
    model = Sequential()
    model.add(Convolution2D(16, 5, 5, activation='relu', input_shape=(28,28, 3)))
    model.add(MaxPooling2D(2, 2))

    model.add(Convolution2D(32, 5, 5, activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))

    model.add(Dense(17, activation='softmax'))
    return model


classifier = create_model()
classifier.load_weights('/home/reshma/admatic/subtraction/model_mnist4.h5')
prev_expr = ''
from keras.preprocessing import image
cap = cv2.VideoCapture(1)


while True:
    ret, frame = cap.read()
    font = cv2.FONT_HERSHEY_SIMPLEX

    #cv2.imshow("orginal capture",frame)
    black_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.blur(black_frame, (5,5))
    #cv2.imshow("gray capture",black_frame)
    ret,t1 = cv2.threshold(blur_frame, 100, 255, cv2.THRESH_BINARY_INV)
    cannied = cv2.Canny(t1,30,150)
    cv2.imshow("boxed canny capture",cannied)
    cv2.imshow("boxed cthresh capture",t1)
    kernel = np.ones((5,5), np.uint8)
    
    # Now we erode
    cannied = cv2.dilate(cannied, kernel, iterations = 4)

    imagenew,contours,hirarchy = cv2.findContours( cannied,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    rframe = frame.copy()
    x_p = []
    val = []
    y_p = []
    if(len(contours)!=0):
        for i in contours:
            x,y,w,h = cv2.boundingRect(i)
            if(w*h>3000 and w*h<70000 and w<2*h ):
            #if((w<180 and w>120 and h<300 and h>240)):
               # cv2.rectangle(rframe,(x,y),(x+w,y+h),(0,255,0),2)
                test_image = t1[y:y+h , x:x+w]
                test_image = cv2.resize(test_image,(int(len(test_image[0]))*5,int(len(test_image))*5),interpolation   = cv2.INTER_AREA)
                #b = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
                #cv2.imshow("boxed canny capture",c)
                ret,t = cv2.threshold(test_image, 127, 255, cv2.THRESH_BINARY_INV)
                kernel = np.ones((5,5), np.uint8)
                d = cv2.erode(t, kernel, iterations = 2)
                h2,w2 = d.shape[:2]
                w1 = w2+100
                h1 = h2+100
                img = np.ones((h1,w1))*255
                img[50:h2+50,50:w2+50]=d[:,:]
                cv2.imwrite('img.jpg',img)
                img  = cv2.imread('img.jpg')
                f = cv2.resize(img,(28,28),interpolation=cv2.INTER_CUBIC)
                f = image.img_to_array(f)
                f = np.expand_dims(f, axis = 0)
                result = classifier.predict(f)
                r_m = result[0].tolist()
                mp = max(r_m)*100
                ans = r_m.index(max(r_m))
                if(ans==0 ):#2==>0==>+
                    cv2.rectangle(rframe,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(rframe,'+ ',(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
                    x_p,y_p,val = insert_val(x_p,y_p,val,'+',x,y)
                elif(ans==1 ):
                    cv2.rectangle(rframe,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(rframe,'- ',(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
                    x_p,y_p,val = insert_val(x_p,y_p,val,'-',x,y)
                elif(ans == 2 ):
                    cv2.rectangle(rframe,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(rframe,'2 ',(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
                    x_p,y_p,val = insert_val(x_p,y_p,val,'0',x,y)
                elif(ans == 3 ):#5==>3==>*
                    cv2.rectangle(rframe,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(rframe,'1',(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
                    x_p,y_p,val = insert_val(x_p,y_p,val,'1',x,y)
                elif(ans == 4):
                    cv2.rectangle(rframe,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(rframe,'2 ',(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
                    x_p,y_p,val = insert_val(x_p,y_p,val,'2',x,y)
                elif(ans == 5):#7==>5==>-
                    cv2.rectangle(rframe,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(rframe,'3',(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
                    x_p,y_p,val = insert_val(x_p,y_p,val,'3',x,y)
                elif(ans == 6):
                    cv2.rectangle(rframe,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(rframe,'4',(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
                    x_p,y_p,val = insert_val(x_p,y_p,val,'4',x,y)
                elif(ans == 7):
                    cv2.rectangle(rframe,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(rframe,'5 ',(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
                    x_p,y_p,val = insert_val(x_p,y_p,val,'5',x,y)
                elif(ans == 8 ):
                    cv2.rectangle(rframe,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(rframe,'6 ',(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
                    x_p,y_p,val = insert_val(x_p,y_p,val,'6',x,y)
                elif(ans == 9  ):#11==>9==>/
                    cv2.rectangle(rframe,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(rframe,'7',(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
                    x_p,y_p,val = insert_val(x_p,y_p,val,'7',x,y)
                elif(ans == 10  ):#11==>9==>/
                    cv2.rectangle(rframe,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(rframe,'8',(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
                    x_p,y_p,val = insert_val(x_p,y_p,val,'8',x,y)
                elif(ans == 11 ):#11==>9==>/
                    cv2.rectangle(rframe,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(rframe,'9',(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
                    x_p,y_p,val = insert_val(x_p,y_p,val,'9',x,y)
                elif(ans == 12 ):#11==>9==>/
                    cv2.rectangle(rframe,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(rframe,'=',(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
                    x_p,y_p,val = insert_val(x_p,y_p,val,'=',x,y)
                elif(ans == 13 ):#11==>9==>/
                    cv2.rectangle(rframe,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(rframe,'/',(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
                    x_p,y_p,val = insert_val(x_p,y_p,val,'/',x,y)
                elif(ans == 14 ):#11==>9==>/
                    cv2.rectangle(rframe,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(rframe,'X',(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
                    x_p,y_p,val = insert_val(x_p,y_p,val,'X',x,y)
                elif(ans == 15):#11==>9==>/
                    cv2.rectangle(rframe,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(rframe,'x',(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
                    x_p,y_p,val = insert_val(x_p,y_p,val,'x',x,y)
                elif(ans == 16):#11==>9==>/
                    cv2.rectangle(rframe,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(rframe,'y',(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
                    x_p,y_p,val = insert_val(x_p,y_p,val,'y',x,y)
                cv2.imshow('frame',rframe)
                if(cv2.waitKey(1)==13):
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)
        try:
                s1,s2 = print_result(x_p,y_p,val)
                #output frame
                out = np.zeros((500,700, 3) , np.uint8)
                
                s1d = 60*(len(s1)-len(s2))
                if(s1d<0):
                    s1d = 0
                s2d = 60*(len(s2)-len(s1))
                if(s2d<0):
                    s2d = 0
                cv2.putText(out,s1,(100+s2d,150), font, 3,(255,0,0),3,cv2.LINE_AA)
                cv2.putText(out,s2,(100+s1d,250), font, 3,(255,0,0),3,cv2.LINE_AA)

                aline  = 50 
                cv2.line(out,(50,275),(650,275),(255,0,0),5)
                cline = 350

                s = max(len(s1),len(s2))*70

                if(len(s1)<len(s2)):
                    s1 = '0'*(len(s2)-len(s1))+s1
                else:
                    s2 = '0'*(len(s1)-len(s2))+s2
                #read strings are reversed to perform subtraction                
                s1 = s1[::-1] 
                s2 = s2[::-1]
                #obtained strings are stored as lists
                str1 = list(s1)
                str2 = list(s2)
                #length of both strings calculated
                n1 = len(s1)
                n2 = len(s2)
                temp = s1
                #comparing every digit in the same column of both numbers
                for i in range(0,n1):
                    if int(str1[i]) < int(str2[i]):
                        str1[i+1] = int(str1[i+1]) - 1
                        
                s1 = "".join(str(str1))
                #taking care of borrow conditions
                for i in range(0,n2):
                    if str2[i] is '0' and str1[i] < '0':
                        borrow = int(str1[i]) + 10
                    elif str2[i] is '0':
                        borrow = int(str1[i])
                    else: 
                        if int(str1[i]) < int(str2[i]):
                            borrow = int(str1[i]) + 10
                        else:
                            borrow = int(str1[i])
                    cv2.putText(out,str(borrow),(s,aline), font, 2,(255,0,0),2,cv2.LINE_AA)
                    ans = int(borrow) - int(s2[i])
                    cv2.putText(out,str(ans),(s,cline), font, 3,(255,0,0),2,cv2.LINE_AA)
                    s = s - 61
            
                
                cv2.imshow('frameanswer',out)
                cv2.imshow('frame',rframe)
        except: 
                pass
        cv2.imshow('frame',rframe)
        if(cv2.waitKey(1)==13):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
```
- Run the following code
```
python3 subtract.py
```
### **Demo video link**

 https://www.dropbox.com/s/2icv2scpo8ckb85/subtraction.mkv?dl=0