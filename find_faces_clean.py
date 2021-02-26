"""

"""

import os
import cv2
# import colorclassification
import filtering

IMAGE_DIRECTORY_NAME = "all rubiks images/simple_rubiks"
EXECUTION_DIRECTORY = os.getcwd()

IMAGE_DIRECTORY = os.path.join(EXECUTION_DIRECTORY,IMAGE_DIRECTORY_NAME)

CONFIDENCE_THRESHOLD = 0.0

orders = {}
images = []

target_width = 760
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
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    inverted_thresholding = filtering.get_inverted_gaussian_thresholding(gray_image)

    erodil = filtering.get_erodil(inverted_thresholding, 2)

    save_sample_image(image, i, "input")
    save_sample_image(erodil, i, "erodilblur")

    # raw_contours = filtering.get_raw_contours(erodil)
    # raw_contour_image = filtering.draw_contours(image, raw_contours)
    # contours = filtering.get_contours(erodil)

    # save_sample_image(filtering.draw_box_contours(image, contours), i, "original")

    # filtering.filterout_byArea(contours)
    # save_sample_image(filtering.draw_box_contours(image, contours), i, "byArea")

    # filtering.filterout_byAspectRatio(contours)
    # save_sample_image(filtering.draw_box_contours(image, contours), i, "byAspectRatio")

    # filtering.filterout_bySolidity(contours,threshold=0.7)
    # save_sample_image(filtering.draw_box_contours(image, contours), i, "bySolidity")

    # filtering.filterout_byEccentricity(contours,threshold=0.7)
    # save_sample_image(filtering.draw_box_contours(image, contours), i, "byEccentricity")

    # filtering.filterout_byModeArea(contours)
    # save_sample_image(filtering.draw_box_contours(image, contours), i, "byModeArea")




