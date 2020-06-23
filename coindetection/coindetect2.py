import cv2
import numpy as np

c = cv2.imread('coin.jpg',0)
cv2.imshow('coin',c)
cv2.waitKey(0)
cv2.destroyAllWindows()
c = cv2.resize(c,(int(len(c[0])/5),int(len(c)/5)),interpolation = cv2.INTER_AREA)
cv2.imshow('coin',c)
cv2.waitKey(0)
cv2.destroyAllWindows()

ker1 = np.ones((3,3))/9
b= cv2.filter2D(c,-1,ker1)

b= cv2.Canny(b,50,150)
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
opening  = cv2.dilate(opening,kernel,iterations = 3)
cv2.imshow('Opening', opening)
cv2.waitKey(0)
cv2.destroyAllWindows()

circles = cv2.HoughCircles(opening, cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
#circles = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT, 1, 10)

circles = np.uint16(np.around(circles))
c1 = c.copy()
p=[]
for i in circles[0,:]:
    if(i[2]>30 and i[2]<60):
        p.append((i[0], i[1]))
        cv2.circle(c1,(i[0], i[1]), i[2], (255, 0, 0), 2)   
        cv2.circle(c1, (i[0], i[1]), 2, (0, 255, 0), 5)

cv2.imshow('detected circles', c1)
cv2.waitKey(0)
cv2.destroyAllWindows()

p

width  = len(c[0])
height = len(c)
print(width,height)
i = 0
c2 = c.copy()
while i<width:
    initial = height
    final = 0
    n = 0
    for j in range(len(p)):
        if(abs(p[j][0]-i)<30):
            if(initial>p[j][1]):
                initial=p[j][1]
            if(final<j):
                final = p[j][1]
            n+=1
    if(n>1):
        cv2.line(c2,(i+30,initial),(i+30,final),(255,0,0),thickness=2)
        i+=80
    else:
        i+=1
cv2.imshow('final', c2)
cv2.imwrite('result.jpg',c2)
cv2.waitKey(0)
cv2.destroyAllWindows()   
        