import sys
import cv2 as cv
import numpy as np


def main(argv):
    ## [load]
    default_file = 'coin.jpg'
    filename = argv[0] if len(argv) > 0 else default_file

    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    ## [load]

    ##orig dimensions
    print('Original Dimensions:',src.shape)

    scale_percent=20
    width=int(src.shape[1]*scale_percent/100)
    height=int(src.shape[0]*scale_percent/100)
    dim=(width,height)

    ##resize
    resized=cv.resize(src,dim,interpolation=cv.INTER_AREA)

    ##print resized shape
    print('Resized dimensions:',resized.shape)

    src=resized

    ## [convert_to_gray]
    # Convert it to gray
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ## [convert_to_gray]
    

    ## [reduce_noise]
    # Reduce the noise to avoid false circle detection
    gray = cv.medianBlur(gray, 5)
    ## [reduce_noise]

    ## [houghcircles]
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 16,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=195)
    ## [houghcircles]
    xcoord=[]
    ycoord=[]
    ## [draw]
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            xcoord.append(i[0])
            ycoord.append(i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)
    ## [draw]
    ## [connect]    
    ctr=0
    j=0
    ymin=5000
    ymax=0
    while j<=570:        
        for k in range(len(xcoord)):
             if abs(xcoord[k]-j) < 5:
                ctr=ctr+1
                if(ycoord[k]<ymin):
                    ymin=ycoord[k]
                elif (ycoord[k]>ymax):
                    ymax=ycoord[k]
        if ctr>2:  
            cv.line(src,(j,ymin),(j,ymax),(255,0,0),3)
            j=j+55
        else:
            j=j+2
    ##[connect]

    ## [display]
    cv.imshow("coins image", src)
    cv.waitKey(0)
    ## [display]
    cv.imwrite('result.jpg',src)
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
