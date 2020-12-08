import cv2
import numpy as np
from matplotlib import pyplot as plt
np.seterr(divide='ignore',invalid='ignore')

r = 1
c = 5
index = 1
def add_img(title, img):
    global index
    plt.subplot(r,c,index)
    plt.imshow(img, None)
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    index+=1

img = cv2.imread("rubiks_cropped/rub0.png")
target_width = 760
aspect_ratio = img.shape[1]/img.shape[0] # w/h
dim = (int(target_width),int(target_width/aspect_ratio))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
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

erosion = np.ones((5,5),np.uint8)
dilation = np.ones((7,7),np.uint8)
def remove_noise(inimg):
    eroded = cv2.erode(invert, erosion)
    dilate = cv2.dilate(eroded, dilation)
    return dilate

nonoise = remove_noise(invert)    
add_img('noise removed', nonoise)

contours, hierarchy  = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
test = cv2.drawContours(color, contours, -1, (0,200,255), 9)
add_img('contours',test)
cv2.imwrite('contours1.png',test)

corners = cv2.goodFeaturesToTrack(nonoise, 10, 0.5, 50)

corner_squares = []
# isolate the best four corners (make the biggest square)
"""
for corner1 in corners:
    for corner2 in corners:
        for corner3 in corners:
            for corner4 in corners:
                # if corner1 == corner2 or corner1 == corner3 or corner1 == corner4 or corner2 == corner3 or corner2 == corner4 or corner3 == corner4:
                #     continue
                # use vectors to represent each side
                x1, y1 = corner1.ravel()
                x2, y2 = corner2.ravel()
                x3, y3 = corner3.ravel()
                x4, y4 = corner4.ravel()
                s1 = (x2 - x1, y2 - y1)
                s2 = (x3 - x2, y3 - y2)
                s3 = (x4 - x3, y4 - y3)
                s4 = (x1 - x4, y1 - y4)
                # calculate dot products to determine if they are rectangles
                mags1 = s1[0]**2 + s1[1]**2
                mags2 = s2[0]**2 + s2[1]**2
                mags3 = s3[0]**2 + s3[1]**2
                mags4 = s4[0]**2 + s4[1]**2
                s1dots2 = s1[0] * s2[0] + s1[1] * s2[1]
                s2dots3 = s2[0] * s3[0] + s2[1] * s3[1]
                s3dots4 = s3[0] * s4[0] + s3[1] * s4[1]
                s4dots1 = s4[0] * s1[0] + s4[1] * s1[1]
                angs1s2 = s1dots2/(mags1*mags2)
                angs2s3 = s2dots3/(mags2*mags3)
                angs3s4 = s3dots4/(mags3*mags4)
                angs4s1 = s4dots1/(mags4*mags1)
                threshold = 0.1
                if abs(angs1s2) < threshold and abs(angs2s3) < threshold and abs(angs3s4) < threshold and abs(angs4s1) < threshold: 
                    corner_squares.append((corner1, corner2, corner3, corner4))
import random
for square in corner_squares:
    col = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    for c in square:
        x, y = c.ravel()
        cv2.circle(img,(x,y),10,col,-1)
"""
for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img,(x,y),10,(36,255,12),-1)
add_img('corners',img)
cv2.imwrite('corners1.png',img)

plt.show()