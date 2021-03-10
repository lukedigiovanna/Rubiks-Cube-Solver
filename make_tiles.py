import os
import numpy as np
import math
import cv2

for i in range(0,28):
    EXECUTION_PATH = os.getcwd()
    IMAGE_PATH = "all rubiks images/for_tiles/"+str(i)+".jpg"
    FULL_IMAGE_PATH = os.path.join(EXECUTION_PATH,IMAGE_PATH)

    # process the image

    erosion_kernel = np.ones((3,3),np.uint8)
    dilation_kernel = np.ones((9,9),np.uint8)

    # read the image
    image = cv2.imread(FULL_IMAGE_PATH)
    target_width = 760
    aspect_ratio = image.shape[1]/image.shape[0] # w/h
    dim = (int(target_width),int(target_width/aspect_ratio))
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    yuv_image[:,:,0] = cv2.equalizeHist(yuv_image[:,:,0])
    yuvrgb_image = cv2.cvtColor(yuv_image,cv2.COLOR_YUV2BGR)

    image_gray = cv2.cvtColor(yuvrgb_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 2)
    thresh_inverted = 255 - thresh

    eroded = cv2.erode(thresh_inverted, erosion_kernel)
    dilated = cv2.dilate(eroded,dilation_kernel)

    blur = cv2.GaussianBlur(dilated, (11,11), 0)

    contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    tile_index = 0

    # handle contours

    for cnt in contours:
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        p1 = box[0]
        p2 = box[1]
        p3 = box[2]
        dx1 = p2[0]-p1[0]
        dy1 = p2[1]-p1[1]
        dx2 = p3[0]-p2[0]
        dy2 = p3[1]-p2[1]
        s1 = math.sqrt(dx1 ** 2 + dy1 ** 2)
        s2 = math.sqrt(dx2 ** 2 + dy2 ** 2)
        
        ratio = s1/s2
        rect_area = s1 * s2
        actual_area = cv2.contourArea(cnt)

        if actual_area < 1000 or rect_area/actual_area > 2 or abs(ratio - 1) > 1:
            continue

        mask = np.zeros(image_gray.shape,np.uint8)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        mean_val = cv2.mean(image,mask=mask)

        # if we got here, the tile is good
        # take points from the box and do perspective transformation
        
        pts1 = np.float32([box[0],box[1],box[3],box[2]])
        pts2 = np.float32([[0,0],[100,0],[0,100],[100,100]])

        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(image, M, (100,100))

        cv2.imwrite(os.path.join(EXECUTION_PATH,"new_tiles/"+str(i)+"_"+str(tile_index)+".jpg"),dst)
        tile_index+=1

