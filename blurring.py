import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('rubiks/rub0.JPG', 0)
plt.subplot(1,2,1)
plt.imshow(img, 'gray')
plt.xticks([]), plt.yticks([])
img = cv2.medianBlur(img, 151)
plt.subplot(1,2,2)
plt.imshow(img, 'gray')
plt.xticks([]), plt.yticks([])

cv2.imwrite("test.png", img)

plt.show()