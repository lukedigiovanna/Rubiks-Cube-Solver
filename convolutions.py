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

img = cv2.imread("rubiks/rub0.JPG", 0)
rescale = 0.2
dim = (int(img.shape[1] * rescale),int(img.shape[0] * rescale))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
add_img('original',img)

# kernel = np.float32([[0, 1, 0],
#                    [1, -4, 1],
#                    [0, 1, 0]])/9
# out = cv2.filter2D(img, -1, kernel)
# add_img('laplacian filter',out)

x = cv2.getGaussianKernel(5, 10)
kernel = x * x.T
out = cv2.filter2D(img, -1, kernel)
add_img('gauss', out)

plt.show()