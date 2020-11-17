import cv2
import numpy as np
from matplotlib import pyplot as plt

r = 1
c = 3
index = 1
def add_img(title, img):
    global index
    plt.subplot(r,c,index)
    plt.imshow(img, None)
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    index+=1

img = cv2.imread("rubiks2/rub0.JPG")
color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
add_img('original',color)

# find edges
edges = cv2.Canny(gray,100,101)
# apply gaussian thresholding
thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# invert
invert = 255 - thresh
add_img('processed', invert)

contours, hierarchy  = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
test = cv2.drawContours(color, contours, -1, (0,200,255), 19)
add_img('contours',test)

plt.show()