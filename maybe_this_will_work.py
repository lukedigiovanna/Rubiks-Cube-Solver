"""

Objective of this attempt:

1. Threshold the image
2. Find regions that are completely enclosed in white regions

"""

import cv2
import numpy as np
import filtering

image = cv2.imread("all rubiks images/rubiks_corners/0.JPG")

image = filtering.resize_image(image, 320)

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
gray_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2GRAY)

threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 2)
dilation_kernel = np.ones((3,3))
dilation = 255-cv2.dilate(threshold, dilation_kernel)

cv2.imwrite("out.png", dilation)