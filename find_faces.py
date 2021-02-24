import os
import cv2
import numpy as np
import math
import keras
import random

EXECUTION_DIRECTORY = os.getcwd()

IMAGE_DIRECTORY_NAME = "all rubiks images/rubiks very simple"
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
            for k in range(j+1,len(inlist)):
                set = (inlist[i],inlist[j],inlist[k])
                sets.append(set)
    return sets
# determined if a set of 3 points are collinear or not
SLOPE_THRESHOLD = 0.1
def is_collinear(set):
    dy = set[0][1] - set[1][1]
    dx = set[0][0] - set[1][0]
    if dx != 0:
        slope1 = dy/dx
    else:
        slope1 = 999
    dy = set[0][1] - set[2][1]
    dx = set[0][0] - set[2][0]
    if dx != 0:
        slope2 = dy/dx
    else:
        slope2 = 999
    dy = set[1][1] - set[2][1]
    dx = set[1][0] - set[2][0]
    if dx != 0:
        slope3 = dy/dx
    else:
        slope3 = 999
    return abs(max(slope1, slope2, slope3) - min(slope1, slope2, slope3)) < SLOPE_THRESHOLD

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

    # yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # yuv_image[:,:,0] = cv2.equalizeHist(yuv_image[:,:,0])
    # yuvrgb_image = cv2.cvtColor(yuv_image,cv2.COLOR_YUV2BGR)
    # yuvrgb_image = image.copy()

    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    image_gray = cv2.cvtColor(yuv_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 2)
    thresh_inverted = 255 - thresh

    eroded = cv2.erode(thresh_inverted, erosion_kernel)
    dilated = cv2.dilate(eroded,dilation_kernel)
    
    blur = cv2.GaussianBlur(dilated, (21,21), 0)

    # edges = cv2.Canny(dilated, 0, 255)
    contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = image.copy()
    cv2.drawContours(contour_image,contours,-1,(0,255,0),3)

    no_area = image.copy()
    no_area_blank = image.copy()
    # no_area_blank[:,:] = [255,255,255]
    output = np.zeros([300,300,3],np.uint8)
    not_square = image.copy()
    not_color = image.copy()
    all_boxes = image.copy()
    masks = np.zeros(image.shape,np.uint8)
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
        dx1 = p2[0]-p1[0]
        dy1 = p2[1]-p1[1]
        dx2 = p3[0]-p2[0]
        dy2 = p3[1]-p2[1]
        s1 = math.sqrt(dx1 ** 2 + dy1 ** 2)
        s2 = math.sqrt(dx2 ** 2 + dy2 ** 2)
        rect_area = s1 * s2
        ratio = s1/s2
        cv2.drawContours(all_boxes,[box],0,(0,255,0),3)
        if cv2.contourArea(cnt) > 1250 and cv2.contourArea(cnt) < 35000:
            cv2.drawContours(no_area,[box],0,(0,255,0),3)
            center = (int(cx),int(cy))
            # centers.append((center,cnt))
            # cv2.drawContours(no_area_blank,[box],0,(0,255,0),3)
            cv2.circle(no_area_blank, center, 3, (255,0,0), 3)
            if abs(ratio - 1) < 10.1:
                cv2.drawContours(not_square,[box],0,(0,255,0), 3)
                mask = np.zeros(image_gray.shape,np.uint8)
                cv2.drawContours(mask, [cnt], 0, 255, -1)
                mean_val = cv2.mean(image,mask=mask)
                raw_color_label, raw_color, raw_confidence = predict_using_raw_rgb(mean_val)
                ratio_color_label, ratio_color, ratio_confidence = predict_using_ratios(mean_val)
                print("RAW", raw_color_label, raw_confidence, "RATIO", ratio_color_label, ratio_confidence)
                if ratio_confidence > CONFIDENCE_THRESHOLD:
                    cv2.drawContours(not_color, [box], 0, (0,255,0), 3)
                    cv2.drawContours(masks, [cnt], 0, ratio_color, -1)
                    cv2.drawContours(masks, [box], 0, (0,255,0), 3)
                    ax += cx
                    ay += cy
                    good_contours.append(cnt)
                    # areas.append(rect_area)
                    areas.append(cv2.contourArea(cnt))
                    aspect_ratios.append(ratio)

    # collinearity determinations
    ax /= len(good_contours)
    ay /= len(good_contours)
    cv2.circle(no_area_blank, (int(ax), int(ay)), 5, (0,0,255), 5)
    centers_only = []
    for c in centers:
        centers_only.append(c[0])
    three_combinations = combinations(centers_only)
    print(len(three_combinations))
    for combination in three_combinations:
        if is_collinear(combination):
            color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            for j in range(2):
                cv2.line(no_area_blank, combination[j], combination[j+1], color, 3)
            # for center in combination:
            #     cv2.circle(no_area_blank, center, 3, color, 20)
    # distances = []
    # for pair in centers:
    #     center = pair[0]
    #     distance = 0
    #     min_val = 999999
    #     min_center = None
    #     for pair2 in centers:
    #         if pair is pair2:
    #             continue
    #         center2 = pair2[0]
    #         # cv2.line(no_area_blank, center, center2, (0,255,0), 2)
    #         dx = center[0] - center2[0]
    #         dy = center[1] - center2[1]
    #         distance = math.sqrt(dx ** 2 + dy ** 2)
    #         if distance < min_val:
    #             min_val = distance
    #             min_center = center2
    #     # cv2.line(no_area_blank, center, min_center, (0, 255, 0), 3)
    #     distances.append((min_val,pair))
    # distances = sorted(distances, key=lambda a: a[0])
    # median_closest = distances[int(len(distances)/2)][0]
    # for j in range(len(distances)):
    #     d = distances[j][0]
    #     if (abs(d - median_closest) < 0.05 * median_closest):
    #         c = distances[j][1][0]
    #         cv2.circle(no_area_blank, c, 5, (255,0,255), 5)


    mode_setting = (areas, 3500)
    # mode_setting = (aspect_ratios, 0.05)
    # calculate the floating point mode of the areas
    epsilon = mode_setting[1] # the range of values accepted within a mode count
    mode_list = mode_setting[0]
    counts = []
    for a in mode_list:
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

    mode_image = image.copy()
    mode_contours = []
    for j, cnt in enumerate(good_contours):
        if abs(mode_list[j] - mode) <= epsilon:
            mode_contours.append(cnt)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)  
            mask = np.zeros(image_gray.shape,np.uint8)
            cv2.drawContours(mask, [cnt], 0, 255, -1)
            mean_val = cv2.mean(image,mask=mask)
            ratio_color_label, ratio_color, ratio_confidence = predict_using_ratios(mean_val)
            cv2.drawContours(mode_image, [box], 0, (0,255,0),3)
    # sort centers based on y position in order to find rows
    centers = []
    for cnt in mode_contours:
        center, radius = cv2.minEnclosingCircle(cnt)
        centers.append((center, cnt))
    centers = sorted(centers,key=lambda ab: ab[0][1])
    if len(centers) >= 9:
        row1 = sorted((centers[0],centers[1],centers[2]), key=lambda ab: ab[0][0])
        row2 = sorted((centers[3],centers[4],centers[5]), key=lambda ab: ab[0][0])
        row3 = sorted((centers[6],centers[7],centers[8]), key=lambda ab: ab[0][0])
        rows = [row1,row2,row3]
        for j in range(3):
            r = rows[j]
            for k in range(3):
                cnt = r[k][1]
                mask = np.zeros(image_gray.shape,np.uint8)
                cv2.drawContours(mask, [cnt], 0, 255, -1)
                mean_val = cv2.mean(image,mask=mask)
                ratio_color_label, ratio_color, ratio_confidence = predict_using_ratios(mean_val)
                cv2.rectangle(output,(k*100,j*100),(k*100+100,j*100+100), ratio_color, -1)
                cv2.rectangle(output,(k*100,j*100),(k*100+100,j*100+100), (0,0,0), 5)

    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"0input.jpg"),image)
    # cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"1yuv.jpg"),yuv_image)
    # cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"2invthresh.jpg"),thresh_inverted)
    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"3erodilblur.jpg"),blur)
    # cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"4contours.jpg"),contour_image)
    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"5contoursquares.jpg"),all_boxes)
    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"6masks.jpg"),masks)
    # cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"6contoursquaresnoarea.jpg"),no_area)
    # cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"6contoursquaresnoareablank.jpg"),no_area_blank)
    # cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"7contoursquaresnosquare.jpg"),not_square)
    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"8contoursquaresnocolor.jpg"),not_color)
    # cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"6contoursquaresmode.jpg"),mode_image)
    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+"7 output.jpg"),output)

    out = image

    print("Finished "+str(i+1)+"/"+str(NUM_IMAGES), str((i+1)/NUM_IMAGES*100))
    
    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"out/"+str(i)+".jpg"), output)

"""
[r, g, r]
[g, b, w]
[b, b, b]


center = face[1][1]


"""