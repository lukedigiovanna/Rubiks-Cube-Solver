import cv2
import numpy as np
from matplotlib import pyplot as plt

r = 1
c = 3
index = 1
def add_img(title, img):
    global index
    plt.subplot(r,c,index)
    plt.imshow(img, 'gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    index+=1

img = cv2.imread("side.png", 0)
add_img('original',img)

edges = cv2.Canny(img,100,101)
add_img('out',edges)

kernel = np.ones((9,9),np.uint8)
dilation = cv2.dilate(edges,kernel,iterations=1)
add_img('dilation',dilation)

plt.show()