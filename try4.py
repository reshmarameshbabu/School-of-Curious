import numpy as np
import cv2

def arrayop(arr,elt,pos):
    cv2.putText(output,"Array:",(100,300),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(output,"Index:",(100,400),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
    x = 250
    y = 300
    p = 230
    q = 270
    c = 250
    d = 370
    n = len(arr)
    for b in range(n):
        cv2.rectangle(output,(p,q),(p + 50,q + 50),(255,0,0),2)
        p += 50
    p = 230
    for i in range(n):
        if count >= i:
            cv2.rectangle(output,(180,335),(900,480),(0,0,0),-1)
            cv2.arrowedLine(output,(c,d),(c,d-30),(0,255,255),2)
            cv2.putText(output,str(arr[i]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
            cv2.putText(output,str(i),(c-10,d+35),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
            x += 50
            c += 50
    if count1 > 0:
        x = 250
        p = 230
        cv2.rectangle(output,(90,335),(900,480),(0,0,0),-1)
        cv2.putText(output,"Element to be inserted: "+str(elt),(100,400),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
        n = len(arr)
        for b in range(n):
            cv2.rectangle(output,(p,q+200),(p + 50,q + 250),(255,0,0),2)
            p += 50
        cv2.putText(output,"Insertion at the end: ",(100,450),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
        for i in range(n):
            cv2.putText(output,str(arr[i]),(x,y+200),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
            x += 50
        cv2.rectangle(output,(90,660),(900,680),(0,0,0),-1)
        arr.append(elt)
        n = len(arr)
        if count1 > 1:
            cv2.rectangle(output,(p,q+200),(p + 50,q + 250),(255,0,0),2)
            cv2.putText(output,str(elt),(x,y+200),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
    if count2 > 0:
        x = 250
        p = 230
        n = len(arr)
        cv2.rectangle(output,(90,335),(900,680),(0,0,0),-1)
        cv2.putText(output,"Element to be inserted: "+str(elt),(100,400),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
        if count1 > 0:
            arr.remove(arr[n-1])
        cv2.putText(output,"Inserting "+str(elt)+" at position "+str(pos),(100,450),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
        arr.insert(int(pos)-1,elt)
        n = len(arr)
        x = 250
        p = 230
        for b in range(n):
            cv2.rectangle(output,(p,q+200),(p + 50,q + 250),(255,0,0),2)
            p += 50
        for i in range(n):
            if count2 >= i:
                cv2.putText(output,str(arr[i]),(x,y+200),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
                x += 50
    if count3 > 0:
        x = 250
        p = 230
        n = len(arr)
        cv2.putText(output,"Element to be deleted: "+str(elt),(100,600),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
        arr.remove(elt)
        n = len(arr)
        for b in range(n):
            cv2.rectangle(output,(p,q+400),(p + 50,q + 450),(255,0,0),2)
            p += 50
        for i in range(n):
            if count3 >= i:
                cv2.putText(output,"Deleting "+str(elt),(100,650),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
                cv2.putText(output,str(a[i]),(x,y+400),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
                x += 50


count = 0
count1 = 0
count2 = 0
count3 = 0
while(True):
    output = np.zeros((1000,1200,3),np.uint8)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        count += 1
    if cv2.waitKey(1) & 0xFF == ord('i'):
        count1 += 1
    if cv2.waitKey(1) & 0xFF == ord('m'):
        count2 += 1
    if cv2.waitKey(1) & 0xFF == ord('d'):
        count3 += 1
    a = [2,5,3,9,7]
    n = len(a)
    elt = 1
    pos = 4
    cv2.arrowedLine(output,(50,65),(75,65),(255,255,255),2)
    cv2.putText(output,"Array is a container which can hold a fix number of items of the same type",(100,75),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
    cv2.arrowedLine(output,(50,100),(75,100),(255,255,255),2)
    cv2.putText(output,"Most of the data structures make use of arrays to implement their algorithms",(100,110),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
    cv2.arrowedLine(output,(50,135),(75,135),(255,255,255),2)
    cv2.putText(output,"Following are the important terms to understand the concept of Array:",(100,140),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
    cv2.arrowedLine(output,(110,165),(140,165),(255,255,255),2)
    cv2.putText(output,"    Element: Each item stored in an array is called an element.",(100,170),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
    cv2.arrowedLine(output,(110,195),(140,195),(255,255,255),2)
    cv2.putText(output,"    Index: Each element in an array has a numerical index,",(100,200),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(output,"             which is used to identify it",(100,230),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
    arrayop(a,elt,pos)

    cv2.imshow('output',output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        