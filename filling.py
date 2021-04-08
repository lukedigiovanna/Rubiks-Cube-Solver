from rubiks_regions import *

dp = "all rubiks images/filling dataset/"
image = Image(dp+"10.JPG")

cv2.imwrite(dp+"output/10image.png",image.image)
cv2.imwrite(dp+"output/10threshold.png",image.threshold)
cv2.imwrite(dp+"output/10mask.png",image.mask)
cv2.imwrite(dp+"output/10contours.png",image.contour_image)