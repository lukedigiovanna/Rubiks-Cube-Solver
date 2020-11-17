import cv2
import numpy as np
from matplotlib import pyplot as plt

r = 2
c = 2
index = 1
def add_img(title, img):
    global index
    plt.subplot(r,c,index)
    plt.imshow(img, 'gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    index+=1

img = cv2.imread("rubiks/rub0.JPG", 0)
add_img('original',img)
ret, simpleThresh = cv2.threshold(img, 177, 255, cv2.THRESH_BINARY)
add_img('simple thresh', simpleThresh)

meanThresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
add_img('mean', meanThresh)

guassianThresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
add_img('gaussian', guassianThresh)

plt.show()