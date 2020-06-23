import cv2
import numpy as np

def lcs(X , Y):
    x = 200
    y = 250
    p = 130
    q = 270 
    m = len(X)
    n = len(Y)
    print(m,n)
    L = [[0 for y in range(m+1)] for x in range(n+1)] 
    for i in range(n+1):
        print(L[i])
    print('\n')
    cv2.putText(output,"The matrix is constructed in the following way:",(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(output,"- if the last characters of both sequences match, value of element in that column is ",(50,75),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(output,"1 + the value of the previous diagonal element",(75,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(output,"- if last characters of both sequences do not match, value of element in that column is",(40,125),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(output,"max(value of element to its left,value of element above it)",(75,150),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
    
    cv2.putText(output,'0',(150,250),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
    cv2.putText(output,'0',(80,300),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
    for i in range(len(X)):
        cv2.putText(output,str(X[i]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
        x += 50
    x = 80
    y = 350
    for i in range(len(Y)):
        cv2.putText(output,str(Y[i]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2,cv2.LINE_AA)
        y += 50
    for i in range(m+1):
        #cv2.rectangle(output,(p,q),(p + 50,q + 50),(255,0,0),2)
        for j in range(n+2):
            cv2.rectangle(output,(p,q+50),(p + 50,q + j*50),(255,100,0),2)
        p += 50
    # Following steps build L[m+1][n+1] in bottom up fashion. Note 
    # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]  
    y = 250
    for i in range(n+1): 
        y += 50
        x = 150
        for j in range(m+1): 
            if i == 0 or j == 0:
                cv2.putText(output,str(L[i][j]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
            elif X[j-1] == Y[i-1]:
                cv2.putText(output,str(L[i][j]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
                cv2.putText(output,str(X[j-1])+" and "+str(Y[i-1])+" are equal",(650,500),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)
                cv2.putText(output,"Element = array["+str(i-1)+"]["+str(j-1)+"] + 1",(650,530),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)
            else:
                cv2.putText(output,str(X[j-1])+" and "+str(Y[i-1])+" are not equal",(650,500),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)
                cv2.putText(output,"Element = Max(array["+str(i-1)+"]["+str(j)+"], array["+str(i)+"["+str(j-1)+"])",(600,530),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)
            cv2.imshow('Output',output)
            if cv2.waitKey(0) & 0xFF == ord('a'):
                cv2.destroyWindow('Output')
            if X[j-1] == Y[i-1]:
                cv2.putText(output,str(L[i][j]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2,cv2.LINE_AA)
            if i == 0 or j == 0: 
                L[i][j] = 0
                cv2.putText(output,str(L[i][j]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
            elif X[j-1] == Y[i-1]: 
                L[i][j] = L[i-1][j-1] + 1
                cv2.putText(output,str(L[i][j]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1]) 
                cv2.putText(output,str(L[i][j]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
            cv2.rectangle(output,(600,400),(1200,800),(0,0,0),-1)
            x += 50
    cv2.imshow('Output',output)
    if cv2.waitKey(0) & 0xFF == ord('a'):
        cv2.destroyWindow('Output')        
    
    print(L[n][m])
    # Following code is used to print LCS 
    index = L[n][m] 
  
    # Create a character array to store the lcs string 
    lcs = [""] * (index+1) 
    lcs[index] = "" 
    # Start from the right-most-bottom-most corner and 
    # one by one store characters in lcs[] 
    i = n 
    j = m 
    while i > 0 and j > 0: 
        # If current character in X and Y are same, then 
        # current character is part of LCS 
        if X[j-1] == Y[i-1]: 
            lcs[index-1] = X[j-1] 
            print(lcs[index-1])
            i-=1
            j-=1
            index-=1
            cv2.arrowedLine(output,(x-50,y-20),(x-80,y-40),(0,255,255),2)
            cv2.circle(output,(x - 40,y - 10),25, (0,255,0), 2)
            cv2.arrowedLine(output,(x-100,y-120),(x-130,y-140),(0,255,255),2)
            cv2.circle(output,(x - 90,y - 110),25, (0,255,0), 2)
            cv2.arrowedLine(output,(x-150,y-220),(x-180,y-240),(0,255,255),2)
            cv2.circle(output,(x - 140,y - 210),25, (0,255,0), 2)
            cv2.arrowedLine(output,(x-200,y-320),(x-230,y-340),(0,255,255),2)
            cv2.circle(output,(x - 190,y - 310),25, (0,255,0), 2)
        # If not same, then find the larger of two and 
        # go in the direction of larger value 
        elif L[i-1][j] > L[i][j-1]: 
            i-=1
            cv2.arrowedLine(output,(x-90,y-70),(x-90,y-90),(0,255,255),2)
            cv2.arrowedLine(output,(x-140,y-170),(x-140,y-190),(0,255,255),2)
            cv2.arrowedLine(output,(x-190,y-270),(x-190,y-290),(0,255,255),2)
        else: 
            j-=1
    s="".join(lcs)
    print ("\nLCS of " + X + " and " + Y + " is " + "".join(lcs))
    cv2.putText(output,"Traversing array from L["+str(m)+"]["+str(n)+"] bottom up,",(100,730),cv2.FONT_HERSHEY_SIMPLEX,0.8,(10,100,255),2,cv2.LINE_AA)
    cv2.putText(output,"- if X[j-1]) and Y[i-1] are equal, include this character as part of LCS.",(100,760),cv2.FONT_HERSHEY_SIMPLEX,0.8,(10,100,255),2,cv2.LINE_AA)
    cv2.putText(output,"- else compare values of array[i-1][j] and array[i][j-1] and ",(100,790),cv2.FONT_HERSHEY_SIMPLEX,0.8,(10,100,255),2,cv2.LINE_AA)
    cv2.putText(output,"                        go in direction of greater value.",(100,820),cv2.FONT_HERSHEY_SIMPLEX,0.8,(10,100,255),2,cv2.LINE_AA)
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
    cv2.putText(output,"LCS of "+str(X)+" and "+str(Y)+" is "+str(s),(100,870),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)
    cv2.putText(output,"Length of LCS: "+str(L[n][m]),(100,900),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2,cv2.LINE_AA)
    cv2.imshow('Output',output)
    if cv2.waitKey(0) & 0xFF == ord('a'):
        cv2.destroyWindow('Output')     
    
count = 0
count1 = 0
while(True):
    output = np.zeros((1000,1200,3),np.uint8)
    X = "AGGTAB"
    Y = "GXTXAYB"
    if cv2.waitKey(1) & 0xFF == ord('s'):
        count1 += 1
    if cv2.waitKey(1) & 0xFF == ord('l'):
        count = 1
    if count == 1:
        lcs(X,Y)
        count = 2
    
    
    cv2.imshow('Output',output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
