import cv2
from matplotlib import pyplot as plt

original_image = cv2.cvtColor(cv2.imread("rubiks2/rub8.jpg"), cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
canny = cv2.Canny(gray, 100, 101)
gaussianThresh = cv2.adaptiveThreshold(canny, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
invert = 255 - gaussianThresh

corners = cv2.goodFeaturesToTrack(invert, 4, 0.5, 50)

for corner in corners:
    x,y = corner.ravel()
    cv2.circle(original_image,(x,y),100,(36,255,12),-1)

plt.subplot(1,1,1)
plt.imshow(original_image, None)
plt.xticks([]), plt.yticks([])

plt.show()
    