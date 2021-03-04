"""

"""

import os
import cv2
import numpy as np
# import colorclassification
import filtering

IMAGE_DIRECTORY_NAME = "all rubiks images/rubiks_corners_no_background"
EXECUTION_DIRECTORY = os.getcwd()

IMAGE_DIRECTORY = os.path.join(EXECUTION_DIRECTORY,IMAGE_DIRECTORY_NAME)

CONFIDENCE_THRESHOLD = 0.0

orders = {}
images = []

target_width = 720
print("Collecting images")
possible_images = os.listdir(IMAGE_DIRECTORY)
for possible_image_name in possible_images:
    if ".JPG" in possible_image_name or ".png" in possible_image_name:
        # print(possible_image_name)
        images.append(filtering.resize_image(cv2.imread(os.path.join(IMAGE_DIRECTORY,possible_image_name)), target_width))

print(str(len(images))+" images collected")

def save_sample_image(image, i, name):
    if i not in orders:
        orders[i] = 0
    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+str(orders[i])+name+".jpg"),image)
    orders[i]+=1

print("Beginning image processing")
for i in range(len(images)):
    image = images[i]
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2GRAY)

    inverted_thresholding = filtering.get_inverted_gaussian_thresholding(cv2.GaussianBlur(gray_image, (51, 51), 0))

    erodil = filtering.get_erodil(inverted_thresholding, 2)

    blur = cv2.GaussianBlur(gray_image, (25,25), 0)
    threshold = filtering.get_inverted_gaussian_thresholding(blur)
    erodil = filtering.get_erodil(threshold, 3)

    raw_contours = filtering.get_raw_contours(erodil)
    contours = filtering.get_contours(erodil)

    filtering.filterout_byArea(contours, lowerbound=250)

    save_sample_image(image,i,"input")
    save_sample_image(erodil,i,"erodil")
    thing = filtering.draw_contours(cv2.cvtColor(erodil,cv2.COLOR_GRAY2BGR), raw_contours)
    save_sample_image(thing,i,"contours")
    save_sample_image(filtering.draw_box_contours(image, contours),i,"boxes")
    save_sample_image(filtering.get_test(image, contours),i,"test")
    save_sample_image(filtering.draw_quad_contours(image, contours),i,"quad")
    save_sample_image(filtering.draw_ellipse_contours(image, contours),i,"ellipses")
    save_sample_image(filtering.get_contour_mask(image, contours, overlay=thing),i,"mask")

    