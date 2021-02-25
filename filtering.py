import cv2
import numpy as np
import math

# returns the inputted image resized to the inputted width while maintaining the original aspect ratio
def resize_image(image, target_width):
    aspect_ratio = image.shape[1]/image.shape[0] # w/h
    dim = (int(target_width),int(target_width/aspect_ratio))
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

# computes the gaussian threshold of the inputted grayscaled image and inverts
# paramters tailored for rubik's cube detection
def get_inverted_gaussian_thresholding(gray_image):
    thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 2)
    return 255 - thresh

# erodes, dilates, and blurs the inputted image with a given kernel size
def get_erodil(input_image, size):
    erosion_kernel = np.ones((size,size),np.uint8)
    dilation_kernel = np.ones((size*2,size*2),np.uint8)
    eroded = cv2.erode(input_image, erosion_kernel)
    dilated = cv2.dilate(eroded,dilation_kernel)
    blur = cv2.GaussianBlur(dilated, (21,21), 0)
    return blur

# gets all detected RAW opencv contours of the inputted image
def get_raw_contours(input_image):
    contours, hierarchy = cv2.findContours(input_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# gets all customly defined contours of the inputted image
def get_contours(input_image):
    raw_contours = get_raw_contours(input_image)
    contours = []
    for cnt in raw_contours:
        contours.append(Contour(cnt))
    return contours

# returns a copy of the inputted image with the inputted contours drawn over it
def draw_contours(image, contours, color=(0,255,0), thickness=3):
    image_copy = image.copy()
    cv2.drawContours(image_copy, contours, -1, color, thickness)
    return image_copy

def draw_box_contours(image, box_contours, color=(0,255,0), thickness=3):
    image_copy = image.copy()
    for cnt in box_contours:
        cv2.drawContours(image_copy,[cnt.box], 0, color, thickness)
    return image_copy

# represents some useful information about a given contour
class Contour:
    def __init__(self, contour):
        self.original_contour = contour
        rect = cv2.minAreaRect(self.original_contour)
        box = cv2.boxPoints(rect)
        self.box = np.int0(box)
        p1 = box[0]
        p2 = box[1]
        p3 = box[2]
        p4 = box[3]
        cx = (p1[0] + p2[0] + p3[0] + p4[0])/4.0
        cy = (p1[1] + p2[1] + p3[1] + p4[1])/4.0
        self.center = (cx, cy)
        dx1 = p2[0]-p1[0]
        dy1 = p2[1]-p1[1]
        dx2 = p3[0]-p2[0]
        dy2 = p3[1]-p2[1]
        s1 = math.sqrt(dx1 ** 2 + dy1 ** 2)
        s2 = math.sqrt(dx2 ** 2 + dy2 ** 2)
        self.rect_area = s1 * s2
        self.aspect_ratio = s1/s2

        
