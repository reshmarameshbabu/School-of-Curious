#importing libraries and dependencies
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense
from queue import Queue

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
x = 80
y = 140
#function for optimal page replacement
def optimal_pagerep(s,capacity):
    occurance = [None for i in range(capacity)]
    f = []
    fault = 0
    p = 80
    q = 210
    c = 100
    d = 240
    x = 110
    y = 110
    for i in range(len(s)):
        if count >= i:
            cv2.rectangle(output,(90,105),(900,145),(0,0,0),-1)
            #pointer to every element of that particular iteration
            cv2.arrowedLine(output,(x,y),(x,y+30),(0,255,255),2)
            q = 210
            #to draw boxes for the frames
            for b in range(capacity):
                    cv2.rectangle(output,(p,q),(p + 50,q + 50),(255,0,0),2)
                    q += 50
            #checking if page is already present
            if s[i] not in f:
                #if number of pages so far less than number of frames, pages just added
                if len(f)<capacity:
                    f.append(s[i])
                else:
                    for a in range(len(f)):
                        #if page needs to be replaced, optimal algorithm implemented
                        #Page that wont be used in the near future is replaced
                        if f[a] not in s[i+1:]:
                            temp = f[a]
                            f[a] = s[i]
                            break
                        else:
                            occurance[a] = s[i+1:].index(f[a])
                    else:
                        f[occurance.index(max(occurance))] = s[i]
                    cv2.rectangle(output,(100,650),(1000,760),(0,0,0),-1)
                    cv2.putText(output,str(s[i])+" enters in place of "+str(temp)+" as "+str(temp)+" is not present in "+str(s[i+1:]),(100,720),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
                    cv2.putText(output, "which is the remaining array",(100,750),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
                fault += 1
                pf = 'PF' #PF signifies page fault
                
            else:
                pf = 'Hit' #page has been found
                cv2.rectangle(output,(100,650),(1000,760),(0,0,0),-1)
                cv2.putText(output,"Found",(100,720),cv2.FONT_HERSHEY_SIMPLEX,0.8,(250,100,100),2,cv2.LINE_AA)

            d = 240
            for k in range(len(f)):
                cv2.putText(output,str(f[k]),(c,d),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
                d += 50
            if pf=='PF':
                cv2.putText(output,str(pf),(c,550),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
            else:
                cv2.putText(output,str(pf),(c,550),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2,cv2.LINE_AA)
            p += 70
            c += 70
            x += 70
    return fault

count = 0
global ARR
import operator
cap = cv2.VideoCapture(1)
st = ""
count = 0
count1 = 0
flag = 0
global ARR
while(True):
    #reading frames from camera
    ret, frame1 = cap.read()
    output = np.zeros((1000,1200,3),np.uint8)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        count += 1
        flag = 1
    if cv2.waitKey(1) & 0xFF == ord('c'):
        count1 += 1
    if flag == 0:
        #splitting the frame thrice in order to read the two strings and the frame capacity
        frame2 = frame1[50:200,100:700]
        frame = frame2.copy()
        frame_new= frame2.copy()
        
        th1 = 63
        ret, img = cv2.threshold(frame, th1, 255, cv2.THRESH_BINARY_INV)
        cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imagenew,contours, hierarchy = cv2.findContours(cvt,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        thisdict = {}
        

        flag=0
        noted_y=0
        mylist = []
        for c in contours:
            (x, y, w, h)= cv2.boundingRect(c)
            if (w>20) or (h>20):
                mylist.append((x,y,w,h))
        
        th2 = 64 #change thresholding of every contour
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
        #print(sort)
        s = ""
        for x in range(0,len(sort)):
            s=s+str(sort[x][1])
        #print(s)

        frame3 = frame1[200:400,100:700]
        frame = frame3.copy()
        frame_new= frame3.copy()
        
        th1 = 63
        ret, img2 = cv2.threshold(frame, th1, 255, cv2.THRESH_BINARY_INV)
        cvt1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        imagenew,contours, hierarchy = cv2.findContours(cvt1 ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        thisdict = {}
        
        flag=0
        noted_y=0
        mylist = []
        for c in contours:
            (x, y, w, h)= cv2.boundingRect(c)
            if (w>20) or (h>20):
                mylist.append((x,y,w,h))
        
        th2 = 64 #change thresholding of every contour
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
            ret, gray1 = cv2.threshold(img1,th2,255,cv2.THRESH_BINARY )
            try:
                im = cv2.resize(gray1, (28,28))
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
        s1 = ""
        for x in range(0,len(sort)):
            s1=s1+str(sort[x][1])
            #cv2.putText(frame1, s, (100,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)  
        s2 = s+","+s1
        print(s2)

        frame4 = frame1[350:600,100:700]
        frame = frame4.copy()
        frame_new= frame4.copy()
        
        th1 = 63
        ret, img1 = cv2.threshold(frame, th1, 255, cv2.THRESH_BINARY_INV)
        
        cvt2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        imagenew,contours, hierarchy = cv2.findContours(cvt2 ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        thisdict = {}
        
        flag=0
        noted_y=0
        mylist = []
        for c in contours:
            (x, y, w, h)= cv2.boundingRect(c)
            if (w>20) or (h>20):
                mylist.append((x,y,w,h))
        
        th2 = 64 #change thresholding of every contour
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
            y=y-10
            x=x-10
            w=w+20
            h=h+20
            cv2.rectangle(frame1,(x,y),(x+w,y+h), (0,0, 255), 2)
            img1 = frame_new[y:y+h, x:x+w]
            ret, gray1 = cv2.threshold(img1,th2,255,cv2.THRESH_BINARY )
            try:
                im = cv2.resize(gray1, (28,28))
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
        s3 = ""
        for x in range(0,len(sort)):
            s3=s3+str(sort[x][1])
            #cv2.putText(frame1, s, (100,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)  
        print(s3)
    try:
        pages = list(s2)
        pages = s2.split(',')
        ARR=pages
        n = len(pages)
        capacity = int(s3)
        x = 100
        y = 170
        cv2.putText(output,"Optimal Page replacement Algorithm",(100,80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)
        for i in range(0,n):
            cv2.putText(output,str(pages[i]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2,cv2.LINE_AA)
            x += 70
        pf = optimal_pagerep(pages,capacity) 
        cv2.putText(output,"No. of frames: "+str(capacity),(100,600),cv2.FONT_HERSHEY_SIMPLEX,0.8,(254,150,0),2,cv2.LINE_AA)
        cv2.putText(output,"No. of page faults: "+str(pf),(100,650),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
     
    except:
        pass
    if count1 >= 1:
            frame5=np.zeros((1200,1000,3),np.uint8)
            cv2.putText(frame5,"def optimal_pagerep(s,capacity):",(50,60),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    occurance = [None for i in range(capacity)]",(50,90),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    f = []",(50,120),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    fault = 0",(50,150),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    for i in range(len(s)):",(50,180),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        if s[i] not in f:",(50,210),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            if len(f)<capacity: ",(50,240),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"                f.append(s[i])",(50,270),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            else:",(50,300),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"                for x in range(len(f)):",(50,330),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"                    if f[x] not in s[i+1:]:",(50,360),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"                        f[x] = s[i]",(50,390),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"                        break",(50,420),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"                    else:",(50,450),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"                        occurance[x] = s[i+1:].index(f[x])",(50,480),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"                else:",(50,510),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"                    f[occurance.index(max(occurance))] = s[i]",(50,540),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            fault += 1",(50,570),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            pf = 'PF' ",(50,600),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        else: ",(50,630),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            pf = 'Hit'",(50,660),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        print(s[i],':',end='')",(50,690),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        for x in f:",(50,720),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"            print(x,end=' ')",(50,750),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"        print(pf)",(50,780),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"    print('Total Page Faults:',fault)",(50,810),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"capacity = 4",(50,840),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"s = "+str(ARR),(50,870),0,0.8,(10,100,255),2)
            cv2.putText(frame5,"optimal_pagerep(s,capacity)",(50,900),0,0.8,(10,100,255),2)
            cv2.imshow("Optimal Page Replacement",frame5)                  
    cv2.imshow('frame1', frame1)
    cv2.imshow('frame', cvt1)
    cv2.imshow('frame2', cvt)
    cv2.imshow('frame4',frame4)
    #cv2.imshow('frame3', img1)
    cv2.imshow('Output',output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break        
cap.release()
#code for code generation of optimal page replacement
f1 = open("optimal_paging.py","w+")
code = '''
def optimal_pagerep(s,capacity):
    occurance = [None for i in range(capacity)]
    f = []
    fault = 0
    for i in range(len(s)):
        if s[i] not in f:
            if len(f)<capacity:
                f.append(s[i])
            else:
                for x in range(len(f)):
                    if f[x] not in s[i+1:]:
                        f[x] = s[i]
                        break
                    else:
                        occurance[x] = s[i+1:].index(f[x])
                else:
                    f[occurance.index(max(occurance))] = s[i]
            fault += 1
            pf = 'PF'
        else:
            pf = 'Hit'
        print(s[i],":",end='')
        for x in f:
            print(x,end=' ')
        
        print(pf)
    print("Total Page Faults:",fault)

capacity = 4
s = '''+str(ARR)+'''
optimal_pagerep(s,capacity)
'''
f1.write(code)
f1.close()


