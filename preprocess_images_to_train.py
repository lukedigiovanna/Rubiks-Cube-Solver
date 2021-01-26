import os
import cv2
import numpy as np
import sys
import math
import keras

EXECUTION_DIRECTORY = os.getcwd()

IMAGE_DIRECTORY_NAME = "simple_rubiks"
IMAGE_DIRECTORY = os.path.join(EXECUTION_DIRECTORY,IMAGE_DIRECTORY_NAME)

NUM_IMAGES = 5

CONFIDENCE_THRESHOLD = 0.6

# helper function to return the maximum index in the given array
def max_index(inlist):
    ind = 0
    max = inlist[ind]
    for i in range(1,len(inlist)):
        if inlist[i] > max:
            max = inlist[i]
            ind = i
    return ind
# helper function to determine whether a prediction can be trusted
def has_confidence(inlist):
    for val in inlist:
        if val > CONFIDENCE_THRESHOLD:
            return True
    return False
def predict_using_raw_rgb(mean):
    predicted = color_model.predict([[mean_val[0],mean_val[1],mean_val[2]]]).tolist()
    max_ind = max_index(predicted[0])
    color_label = color_labels[max_ind]
    color = color_values[max_ind]
    confidence = predicted[0][max_ind]
    return color_label, color, confidence

def predict_using_ratios(mean):
    rg = mean[2]/mean[1]
    rb = mean[2]/mean[0]
    gb = mean[1]/mean[0]
    predicted = color_ratio_model.predict([[rg,rb,gb]]).tolist()
    max_ind = max_index(predicted[0])
    color_label = color_labels[max_ind]
    color = color_values[max_ind]
    confidence = predicted[0][max_ind]
    return color_label, color, confidence

# load color classification model
color_model = keras.models.load_model(os.path.join(EXECUTION_DIRECTORY,"colorclassification.h5"))
color_ratio_model = keras.models.load_model(os.path.join(EXECUTION_DIRECTORY,"colorratioclassification.h5"))

# stores the class labels respective to the model prediction output
color_labels = ('blue','green','orange','red','white','yellow')
color_values = ((255,0,0),(0,255,0),(0,125,255),(0,0,255),(255,255,255),(0,255,255))

erosion_kernel = np.ones((3,3),np.uint8)
dilation_kernel = np.ones((6,6),np.uint8)

for i in range(NUM_IMAGES):
    # read the image
    image = cv2.imread(os.path.join(IMAGE_DIRECTORY,str(i)+".jpg"))
    target_width = 760
    aspect_ratio = image.shape[1]/image.shape[0] # w/h
    dim = (int(target_width),int(target_width/aspect_ratio))
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    yuv_image[:,:,0] = cv2.equalizeHist(yuv_image[:,:,0])
    yuvrgb_image = cv2.cvtColor(yuv_image,cv2.COLOR_YUV2BGR)

    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"1yuv.jpg"),yuvrgb_image)

    image_gray = cv2.cvtColor(yuvrgb_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 2)
    thresh_inverted = 255 - thresh

    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"2invthresh.jpg"),thresh_inverted)

    eroded = cv2.erode(thresh_inverted, erosion_kernel)
    dilated = cv2.dilate(eroded,dilation_kernel)
    
    blur = cv2.GaussianBlur(dilated, (17,17), 0)

    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"3erodilblur.jpg"),blur)

    # edges = cv2.Canny(dilated, 0, 255)
    contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = image.copy()
    cv2.drawContours(contour_image,contours,-1,(0,255,0),3)
    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"4contours.jpg"),contour_image)

    no_area = image.copy()
    not_square = image.copy()
    not_color = image.copy()
    all_boxes = image.copy()
    for cnt in contours:
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
        s1 = dx1 ** 2 + dy1 ** 2
        s2 = dx2 ** 2 + dy2 ** 2
        ratio = math.sqrt(s1/s2)
        cv2.drawContours(all_boxes,[box],0,(0,255,0),3)
        if cv2.contourArea(cnt) > 1250 and cv2.contourArea(cnt) < 20000:
            cv2.drawContours(no_area,[box],0,(0,255,0),3)
            if abs(ratio - 1) < 0.3:
                cv2.drawContours(not_square,[box],0,(0,255,0), 3)
                mask = np.zeros(image_gray.shape,np.uint8)
                cv2.drawContours(mask, [cnt], 0, 255, -1)
                mean_val = cv2.mean(image,mask=mask)
                raw_color_label, raw_color, raw_confidence = predict_using_raw_rgb(mean_val)
                ratio_color_label, ratio_color, ratio_confidence = predict_using_ratios(mean_val)
                print("RAW", raw_color_label, raw_confidence, "RATIO", ratio_color_label, ratio_confidence)
                if ratio_confidence > CONFIDENCE_THRESHOLD:
                    cv2.drawContours(not_color, [box], 0, ratio_color, 3)

    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"5contoursquares.jpg"),all_boxes)
    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"6contoursquaresnoarea.jpg"),no_area)
    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"7contoursquaresnosquare.jpg"),not_square)
    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"8contoursquaresnocolor.jpg"),not_color)

    """
    areas = []
    possible_contours = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        rectangles = image.copy()
        cv2.drawContours(rectangles,[box],0,(0,255,0),3)
        cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"5contoursquares.jpg"),rectangles)

        # box is a 2D array of points. 
        # what we can do is find the dimensions of the box given these points
        # only need to get the first two sides
        p1 = box[0]
        p2 = box[1]
        p3 = box[2]
        dx1 = p2[0]-p1[0]
        dy1 = p2[1]-p1[1]
        dx2 = p3[0]-p2[0]
        dy2 = p3[1]-p2[1]
        s1 = dx1 ** 2 + dy1 ** 2
        s2 = dx2 ** 2 + dy2 ** 2
        
        ratio = math.sqrt(s1/s2)
        rect_area = s1 * s2
        actual_area = cv2.contourArea(cnt)

        if actual_area < 1000 or rect_area/actual_area > 2 or abs(ratio - 1) > 1:
            continue
        else:
            areas.append(cv2.contourArea(cnt))
            possible_contours.append(cnt)
    epsilon = 750
    counts = []
    for a in areas:
        for j, (b, c) in enumerate(counts):
            if -epsilon <= a - b <= epsilon:
                counts[j] = (b, c + 1)
                break
        else:
            counts.append((a,1))
    possible_modes = sorted(counts,key=lambda ab: (ab[1],ab[0]))
    possible_modes.reverse()
    print(possible_modes)
    # for possible_mode in possible_modes:
    mode = -1
    if len(possible_modes) > 0:
        mode = possible_modes[0][0]

    # cv2.drawContours(image, contours, -1, (0,255,255), 3)
    for cnt in possible_contours:
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        actual_area = cv2.contourArea(cnt)
        if mode < 0 or -epsilon <= actual_area - mode <= epsilon:
            mask = np.zeros(image_gray.shape,np.uint8)
            cv2.drawContours(mask, [cnt], 0, 255, -1)
            mean_val = cv2.mean(image,mask=mask)
            predicted = color_model.predict([[mean_val[0],mean_val[1],mean_val[2]]]).tolist()
            max_ind = max_index(predicted[0])
            color_label = color_labels[max_ind]
            color = color_values[max_ind]
            if not has_confidence(predicted[0]):
                color = (0,0,0)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(image,[box],0,color,2)
            cv2.circle(image,center,3,(0,0,0),5)
    """

    out = image

    print("Finished "+str(i+1)+"/"+str(NUM_IMAGES), str((i+1)/NUM_IMAGES*100))
    
    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"out/"+str(i)+".jpg"), out)