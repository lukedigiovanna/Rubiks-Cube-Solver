import os
import cv2
import numpy as np
import sys
import math
import keras

EXECUTION_DIRECTORY = os.getcwd()

IMAGE_DIRECTORY_NAME = "rubiks very simple"
IMAGE_DIRECTORY = os.path.join(EXECUTION_DIRECTORY,IMAGE_DIRECTORY_NAME)

NUM_IMAGES = 6

CONFIDENCE_THRESHOLD = 0.0

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
# predicts the color label of a mean color using raw rgb color model
def predict_using_raw_rgb(mean):
    predicted = color_model.predict([[mean_val[0]/255,mean_val[1]/255,mean_val[2]/255]]).tolist()
    max_ind = max_index(predicted[0])
    color_label = color_labels[max_ind]
    color = color_values[max_ind]
    confidence = predicted[0][max_ind]
    return color_label, color, confidence
# predicts the color label of a mean color using ratios of red green and blue color model
# this model is recommended over the RGB model as it is more accurate
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

# helper function to gather all sets of 3 points in the process of determining collinearity
def combinations(inlist):
    sets = []
    for i in range(len(inlist)-2):
        for j in range(i+1,len(inlist)-1):
            set = (inlist[i],inlist[j],inlist[j+1])
            sets.append(set)
    return sets
# determined if a set of 3 points are collinear or not
SLOPE_THRESHOLD = 0.1
def is_collinear(set):
    slope1 = (set[0][1] - set[1][1])/(set[0][0] - set[1][0])
    slope2 = (set[0][1] - set[2][1])/(set[0][0] - set[2][0])
    return abs(slope1 - slope2) < SLOPE_THRESHOLD

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
    yuvrgb_image = image.copy()

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
    no_area_blank = image.copy()
    no_area_blank[:,:] = [255,255,255]
    not_square = image.copy()
    not_color = image.copy()
    all_boxes = image.copy()
    good_contours = []
    areas = []
    aspect_ratios = []
    ax = 0
    ay = 0
    centers = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        p1 = box[0]
        p2 = box[1]
        p3 = box[2]
        p4 = box[3]
        cx = (p1[0] + p2[0] + p3[0] + p4[0])/4.0
        cy = (p1[1] + p2[1] + p3[1] + p4[1])/4.0
        ax += cx
        ay += cy
        dx1 = p2[0]-p1[0]
        dy1 = p2[1]-p1[1]
        dx2 = p3[0]-p2[0]
        dy2 = p3[1]-p2[1]
        s1 = math.sqrt(dx1 ** 2 + dy1 ** 2)
        s2 = math.sqrt(dx2 ** 2 + dy2 ** 2)
        rect_area = s1 * s2
        ratio = math.sqrt(s1/s2)
        cv2.drawContours(all_boxes,[box],0,(0,255,0),3)
        if cv2.contourArea(cnt) > 1250 and cv2.contourArea(cnt) < 25000:
            cv2.drawContours(no_area,[box],0,(0,255,0),3)
            center = (int(cx),int(cy))
            centers.append(center)
            # cv2.drawContours(no_area_blank,[box],0,(0,255,0),3)
            cv2.circle(no_area_blank, center, 3, (255,0,0), 3)
            if abs(ratio - 1) < 1000.3:
                cv2.drawContours(not_square,[box],0,(0,255,0), 3)
                mask = np.zeros(image_gray.shape,np.uint8)
                cv2.drawContours(mask, [cnt], 0, 255, -1)
                mean_val = cv2.mean(image,mask=mask)
                raw_color_label, raw_color, raw_confidence = predict_using_raw_rgb(mean_val)
                ratio_color_label, ratio_color, ratio_confidence = predict_using_ratios(mean_val)
                print("RAW", raw_color_label, raw_confidence, "RATIO", ratio_color_label, ratio_confidence)
                if ratio_confidence > CONFIDENCE_THRESHOLD:
                    cv2.drawContours(not_color, [box], 0, (0,255,0), 3)
                    good_contours.append(cnt)
                    # areas.append(rect_area)
                    areas.append(cv2.contourArea(cnt))
                    aspect_ratios.append(ratio)
    ax /= len(contours)
    ay /= len(contours)
    cv2.circle(no_area_blank, (int(ax), int(ay)), 5, (0,0,255), 5)
    distances = []
    for center in centers:
        distance = 0
        for center2 in centers:
            if center is center2:
                continue
            cv2.line(no_area_blank, center, center2, (0,255,0), 2)
            dx = center[0] - center2[0]
            dy = center[1] - center2[1]
            distance += dx ** 2 + dy ** 2
        distances.append((distance,center))
    distances = sorted(distances, key=lambda a: a[0])
    for j in range(min(9,len(distances))):
        c = distances[j][1]
        cv2.circle(no_area_blank, c, 5, (255,0,255), 5)
    
    
    # mode_setting = (areas, 2500)
    # mode_setting = (aspect_ratios, 0.05)
    # # calculate the floating point mode of the areas
    # epsilon = mode_setting[1] # the range of values accepted within a mode count
    # mode_list = mode_setting[0]
    # counts = []
    # for a in mode_list:
    #     for j, (b, c) in enumerate(counts):
    #         if -epsilon <= a - b <= epsilon:
    #             counts[j] = (b, c + 1)
    #             break
    #     else:
    #         counts.append((a,1))
    # possible_modes = sorted(counts,key=lambda ab: (ab[1],ab[0]))
    # possible_modes.reverse()
    # print(possible_modes)
    # # for possible_mode in possible_modes:
    # mode = -1
    # if len(possible_modes) > 0:
    #     mode = possible_modes[0][0]

    # mode_image = image.copy()
    # mode_contours = []
    # for j, cnt in enumerate(good_contours):
    #     if abs(mode_list[j] - mode) <= epsilon:
    #         mode_contours.append(cnt)
    #         rect = cv2.minAreaRect(cnt)
    #         box = cv2.boxPoints(rect)
    #         box = np.int0(box)  
    #         cv2.drawContours(mode_image, [box], 0, (0,255,0),3)

    # sets = combinations(good_contours)
    # collinear_contours = []
    # for set in sets:
    #     points = []
    #     for cnt in set:
    #         (x,y), radius = cv2.minEnclosingCircle(cnt)
    #         points.append((x,y))
    #     if is_collinear(points):
    #         for cnt in set:
    #             # if cnt not in collinear_contours:
    #             collinear_contours.append(cnt)
    # 
    # for cnt in collinear_contours:
    #     (x,y), radius = cv2.minEnclosingCircle(cnt)
    #     center = (x, y)
    #     cv2.circle(not_color, (int(x),int(y)), 1, (255,0,0), 5)

    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"5contoursquares.jpg"),all_boxes)
    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"6contoursquaresnoarea.jpg"),no_area)
    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"6contoursquaresnoareablank.jpg"),no_area_blank)
    # cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"7contoursquaresnosquare.jpg"),not_square)
    # cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"8contoursquaresnocolor.jpg"),not_color)
    # cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"9contoursquaresmode.jpg"),mode_image)

    out = image

    print("Finished "+str(i+1)+"/"+str(NUM_IMAGES), str((i+1)/NUM_IMAGES*100))
    
    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"out/"+str(i)+".jpg"), out)