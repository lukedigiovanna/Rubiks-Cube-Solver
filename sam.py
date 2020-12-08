import cv2
import numpy as np
from matplotlib import pyplot as plt

r = 2
c = 5
index = 1
def add_img(title, img):
    global index
    plt.subplot(r,c,index)
    plt.imshow(img, 'gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    index+=1

def get_sift(inimg):
   # gray = cv2.cvtColor(inimg,cv2.COLOR_BGR2GRAY)
    #i = inimg
    return inimg
    # sift = cv2.SIFT_create()
    # kp = sift.detect(inimg,None)
    # return cv2.drawKeypoints(inimg, kp, inimg,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img = cv2.imread("rubiks_cropped/rub0.png", 0)
width = 720
dim = (width,int(img.shape[1]/img.shape[0] * width))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
add_img('original',img)

edges = cv2.Canny(img,100,101)
add_img('edges',edges)

gaussianThresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
add_img('gaussian', gaussianThresh)

invert = 255 - gaussianThresh                
print(invert)
add_img('invert',invert)

kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(invert,kernel,iterations=1)
add_img('dilation',dilation)

original_sift = get_sift(img)
add_img('original sift',original_sift)
add_img('edges sift', get_sift(edges))
add_img('gaussian sift', get_sift(gaussianThresh))
add_img('invert sift', get_sift(invert))
add_img('dilation sift', get_sift(dilation))

plt.show()