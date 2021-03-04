"""

Objective of this attempt:

1. Threshold the image
2. Find regions that are completely enclosed in white regions

"""

import cv2
import numpy as np
import colorclassification

image = cv2.imread("all rubiks images/rubiks_corners/0.JPG")

target_width = 180

aspect_ratio = image.shape[1]/image.shape[0] # w/h
dim = (int(target_width),int(target_width/aspect_ratio))
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
gray_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2GRAY)

threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2)
dilation_kernel = np.ones((3,3))
threshold = 255-cv2.dilate(threshold, dilation_kernel)

# 
# copy = np.zeros(dilation.shape,np.uint8)
# for r in range(1, len(dilation)-1):
#     for c in range(1, len(dilation[0])-1):
#         # check all directly adjacent sides
#         if dilation[r][c] == 0 and (dilation[r-1][c] == 255 or dilation[r+1][c] == 255 or dilation[r][c-1] == 255 or dilation[r][c+1] == 255):
#             copy[r][c] = 255

cache = []

row_i = 0
height, width = threshold.shape

# returns a list of the pixel coordinates that make up the region
def find_region(x, y, region):
    point = (x, y)
    if x < 0 or x >= width or y < 0 or y >= height or point in cache or threshold[y][x] == 0:
        return []
    else:
        region.append(point)
        cache.append(point)
        find_region(x + 1, y, region)
        find_region(x - 1, y, region)
        find_region(x, y + 1, region)
        find_region(x, y - 1, region)

def average_color(region):
    mean = [0,0,0]
    for pixel in region:
        color = image[pixel[1]][pixel[0]]
        for i in range(3):
            mean[i] += color[i]
    for i in range(3):
        mean[i] /= len(region)
    return mean

def predict_color(region):
    mean = average_color(region)
    label, color, confidence = colorclassification.predict_using_ratios(mean)
    if confidence < 0.6:
        return (255,0,255)
    else:
        return color

blank = np.zeros(image.shape, np.uint8)

while row_i < height:
    col_i = 0
    while col_i < width:
        region = []
        find_region(col_i, row_i, region)
        if len(region) <= 50:
            col_i += 1
            continue
        print(len(region))
        color = predict_color(region)
        for p in region:
            blank[p[1]][p[0]] = color
        col_i += 1
    row_i += 1

cv2.imwrite("threshold.png", blank)