import cv2
import filtering

test_image = filtering.resize_image(cv2.imread("all rubiks images/rubiks_corners/0.JPG"), 760)
# threshold = 100
# for row in test_image:
#     for col in row:
#         col[0] = col[0] if col[0] >= threshold else 0
#         col[1] = col[1] if col[1] >= threshold else 0
#         col[2] = col[2] if col[2] >= threshold else 0

red = test_image[:,:,0]
green = test_image[:,:,1]
blue = test_image[:,:,2]

# red = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
# green = cv2.adaptiveThreshold(green, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
# blue = cv2.adaptiveThreshold(blue, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)

ksize = 21
red = cv2.GaussianBlur(red, (ksize, ksize), 0)
green = cv2.GaussianBlur(green, (ksize, ksize), 0)
blue = cv2.GaussianBlur(blue, (ksize, ksize), 0)

full = red + green + blue

gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)

filtered = cv2.adaptiveThreshold(cv2.cvtColor(test_image - cv2.GaussianBlur(test_image, (51,51), 5) + 150, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)

cv2.imwrite("out.png",full)
cv2.imwrite("outbaljkef.png",cv2.GaussianBlur(gray, (51,51),0))
cv2.imwrite("red.png",red)
cv2.imwrite("green.png",green)
cv2.imwrite("blue.png",blue)

print("done")