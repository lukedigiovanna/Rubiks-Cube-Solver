import cv2
import numpy as np
from matplotlib import pyplot as plt

r = 3
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
add_img('global thresh', simpleThresh)

add_img('original',img)
ret2, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
add_img('otsu', otsu)

blurred = cv2.GaussianBlur(img, (17,17), 0)
ret3, gotsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
add_img('gaussian + otsu', gotsu)

plt.show()