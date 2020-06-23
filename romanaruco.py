import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import math
from sympy import *

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

cap = cv2.VideoCapture(1)

aruco_list = {}
ar_list = {}
number_list={}
tx2=0
x1=0;y1=0
temp1=0
z=0

ret, frame=cap.read()
vid_writer = cv2.VideoWriter('output_roman.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

while True:
    ret, frame = cap.read()
    font = cv2.FONT_HERSHEY_SIMPLEX
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    arucoParameters =  aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=arucoParameters)
    #print('corner=',corners,'ids=',ids,'t=',t)
    
    number_list={}
    centre_list = {}
    final_list={}
    orig_ids={}
    origids_y={}
    exp=[]
    split_list=[]
    n=1
    if len(corners)==3:
    	lis=[]
    if np.all(ids != None):
        j = 0
        #print(len(corners))
        for k in range(len(corners)):
            temp_1 = corners[k]
            temp_1 = temp_1[0]
            temp_2 = ids[k]
            temp_2 = temp_2[0]
            aruco_list[temp_2] = temp_1
            ar_list[j] = temp_1
            j = j + 1
        
 
        
        key_list = aruco_list.keys()
        font = cv2.FONT_HERSHEY_SIMPLEX
        for key in key_list:
            dict_entry = aruco_list[key]    
            centre = dict_entry[0] + dict_entry[1] + dict_entry[2] + dict_entry[3]
            centre[:] = [int(x / 4) for x in centre]
            centre_list[key] = tuple(centre)
            

        for i in range(len(corners)):
            corner = (int(ar_list[i][0][0]),int(ar_list[i][0][1]))
            #print(ids[i][0],ar_list[i][0][0]) 
            number_list[ids[i][0]]= int(ar_list[i][0][0])

            final_list[ids[i][0]]= centre_list[ids[i][0]]



            if ids[i][0]==1:
               cv2.putText(frame,'I', centre_list[ids[i][0]], font, 1, (0,0,255), 2, cv2.LINE_AA)   
               orig_ids[centre_list[ids[i][0]][0]]='I'
               origids_y[centre_list[ids[i][0]][1]]='I'



            
            if ids[i][0]==2:
               cv2.putText(frame,'I', centre_list[ids[i][0]], font, 1, (0,0,255), 2, cv2.LINE_AA)
               orig_ids[centre_list[ids[i][0]][0]]='I'
               origids_y[centre_list[ids[i][0]][1]]='I'


            if ids[i][0]==3:
               cv2.putText(frame,'I', centre_list[ids[i][0]], font, 1, (0,0,255), 2, cv2.LINE_AA)
               orig_ids[centre_list[ids[i][0]][0]]='I'
               origids_y[centre_list[ids[i][0]][1]]='I'

            if ids[i][0]==4:
               cv2.putText(frame,'I', centre_list[ids[i][0]], font, 1, (0,0,255), 2, cv2.LINE_AA)
               orig_ids[centre_list[ids[i][0]][0]]='I'
               origids_y[centre_list[ids[i][0]][1]]='I'


            if ids[i][0]==5:
               cv2.putText(frame,'V', centre_list[ids[i][0]], font, 1, (0,0,255), 2, cv2.LINE_AA)
               orig_ids[centre_list[ids[i][0]][0]]='V'
               origids_y[centre_list[ids[i][0]][1]]='V'
            
            if ids[i][0]==6:
               cv2.putText(frame,'V', centre_list[ids[i][0]], font, 1, (0,0,255), 2, cv2.LINE_AA)
               orig_ids[centre_list[ids[i][0]][0]]='V'
               origids_y[centre_list[ids[i][0]][1]]='V'

            if ids[i][0]==7:
               cv2.putText(frame,'X', centre_list[ids[i][0]], font, 1, (0,0,255), 2, cv2.LINE_AA)
               orig_ids[centre_list[ids[i][0]][0]]='X'
               origids_y[centre_list[ids[i][0]][1]]='X'

            if ids[i][0] not in [1,2,3,4,5,6,7]:
               cv2.putText(frame,str(ids[i][0]), centre_list[ids[i][0]], font, 1, (0,255,0), 2, cv2.LINE_AA)
               orig_ids[centre_list[ids[i][0]][0]]=str(ids[i][0])
               origids_y[centre_list[ids[i][0]][1]]=str(ids[i][0])
               

        
        final_ids_list=final_list.keys()

        number_list= sorted(number_list.items(), key = lambda kv:(kv[1], kv[0]))

        orig_ids_sort=sorted(orig_ids.keys())
        #print(orig_ids.items())
        origids_ysort=sorted(origids_y.keys())

        for i in orig_ids_sort:
                exp.append(orig_ids[i])
                split_list.append(orig_ids[i])

     
        fullexp=("".join(exp))

        ans=romanToDecimal(str(fullexp))
        #print(fullexp)
        cv2.putText(frame,'in roman '+str(fullexp) +' ==> '+str(ans), (30,40), font, 1, (0,0,255), 2, cv2.LINE_AA)
        
        
        #print(ans)
 

        
        aruco_list.clear()
        display = aruco.drawDetectedMarkers(frame, corners)
        display = np.array(display)
    else:
        display = frame
    vid_writer.write(frame)
    cv2.imshow('Display',display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()