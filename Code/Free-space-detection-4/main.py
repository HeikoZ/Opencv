import cv2 as cv
import os
from kmeans import kemeans_roi
from Network import getimgfromjson
import time



# Two categries
#cap = cv.VideoCapture("../RAW_data/Test_video/out.avi")
cap = cv.VideoCapture("./RAW_data/Test_video/out.avi")
# get width and height
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# creat video out object
#out = cv.VideoWriter('../RAW_data/demo/demo_video.avi',
#                     cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

out = cv.VideoWriter('./RAW_data/demo/demo_video.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

## Set a update frequency of detection results.
# ten frames remove noise
frequency = 20

# create mask
freespacerange = []
empty_out = []
full_out = []

if(cap.isOpened()==False):
    print("can not read video")

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:

        #computer time
        start = time.time()

        if(frequency == 20):
            """
            # midfreespace
            freespacerange = kemeans_roi(frame)
   """
            #free space with line
            empty_out,full_out = getimgfromjson(frame)

            #reset frequency
            frequency = 0
        else:
            frequency += 1

        for i in range(len(empty_out)):
            cv.polylines(frame, [empty_out[i]], True, (0, 255, 0))
        for i in range(len(full_out)):
            cv.polylines(frame, [full_out[i]], True, (0, 0, 255))

        ## Set the location of putting text.
        #put text in image
        cv.putText(frame,"Parking_space_with_line: "+str(len(empty_out)),(1050,580),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        #endtime
        end = time.time()
        print("used:",end-start)

        cv.imshow("frame", frame)
        cv.waitKey(25)
        out.write(frame)

    # Break the loop
    else:
        break

