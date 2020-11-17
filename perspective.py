import cv2
import numpy as np
from matplotlib import pyplot as plt

r = 1
c = 2
index = 1
def add_img(title, img):
    global index
    plt.subplot(r,c,index)
    plt.imshow(img, 'gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    index+=1

img = cv2.imread("rubiks/rub0.JPG", 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
add_img('original',img)

pts1 = np.float32([[801,1324],[1952,1595],[859,2184],[1820,2432]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(300,300))

add_img('out',dst)
cv2.imwrite("side.png",dst)

plt.show()