from rubiks_regions import *

dp = "all rubiks images/rubiks_corners_3/"
one = Image(dp+"0.JPG")

cv2.imwrite("0threshold.png",one.threshold)