"""

"""

import os
import cv2
from numpy.core.fromnumeric import resize
import colorclassification
import filtering

IMAGE_DIRECTORY_NAME = "all rubiks images/rubiks very simple"
EXECUTION_DIRECTORY = os.getcwd()

IMAGE_DIRECTORY = os.path.join(EXECUTION_DIRECTORY,IMAGE_DIRECTORY_NAME)

CONFIDENCE_THRESHOLD = 0.0

orders = {}

def save_sample_image(image, i, name):
    if i not in orders:
        orders[i] = 0
    cv2.imwrite(os.path.join(IMAGE_DIRECTORY,"sample_images/"+str(i)+str(orders[i])+name+".jpg"),image)
    orders[i]+=1

images = []

possible_images = os.listdir(IMAGE_DIRECTORY)
for possible_image_name in possible_images:
    if ".JPG" in possible_image_name:
        images.append(cv2.imread(os.path.join(IMAGE_DIRECTORY,possible_image_name)))

target_width = 760
for i in range(len(images)):
    raw_image = images[i]
    resized_image = filtering.resize_image(raw_image, target_width)
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    inverted_thresholding = filtering.get_inverted_gaussian_thresholding(gray_image)

    erodil = filtering.get_erodil(inverted_thresholding, 3)

    raw_contours = filtering.get_raw_contours(erodil)
    raw_contour_image = filtering.draw_contours(resized_image, raw_contours)
    contours = filtering.get_contours(erodil)

    box_contour_image = filtering.draw_box_contours(resized_image, contours)

    save_sample_image(raw_image, i, "input")
    save_sample_image(hsv_image, i, "hsv")
    save_sample_image(inverted_thresholding, i, "invertedThresholding")
    save_sample_image(erodil, i, "erodil")
    save_sample_image(raw_contour_image, i, "originalContours")
    save_sample_image(box_contour_image, i, "originalBoxContours")





