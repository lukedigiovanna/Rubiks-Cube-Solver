import cv2
import numpy as np
import math
import colorclassification

# returns the inputted image resized to the inputted width while maintaining the original aspect ratio
def resize_image(image, target_width):
    aspect_ratio = image.shape[1]/image.shape[0] # w/h
    dim = (int(target_width),int(target_width/aspect_ratio))
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

def get_rgb_thresholded(rgb_image, threshold):
    image = rgb_image.copy()
    for row in image:
        for col in row:
            val = 255 if col[0] >= threshold or col[1] >= threshold or col[2] >= threshold else 0
            col[0] = val
            col[1] = val
            col[2] = val
    return image

# computes the gaussian threshold of the inputted grayscaled image and inverts
# paramters tailored for rubik's cube detection
def get_inverted_gaussian_thresholding(gray_image):
    thresh = 255-cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
    return thresh

def get_edges(gray_image):
    return cv2.Canny(gray_image, 200, 255)

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

def draw_box_contours(image, contours, color=(0,255,0), thickness=3):
    image_copy = image.copy()
    for cnt in contours:
        cv2.drawContours(image_copy,[cnt.box], 0, color, thickness)
    return image_copy

def draw_ellipse_contours(image, contours, color=(0,255,0), thickness=3):
    image_copy = image.copy()
    for cnt in contours:
        cv2.ellipse(image_copy,cnt.center,cnt.axes,cnt.angle,0,360,color,thickness=thickness)
    return image_copy

def draw_quad_contours(image, contours, color=(0,255,0), thickness=3):
    image_copy = image.copy()
    for cnt in contours:
        image_copy = cv2.line(image_copy, cnt.leftmost, cnt.topmost, color=color, thickness=thickness)
        image_copy = cv2.line(image_copy, cnt.topmost, cnt.rightmost, color=color, thickness=thickness)
        image_copy = cv2.line(image_copy, cnt.rightmost, cnt.bottommost, color=color, thickness=thickness)
        image_copy = cv2.line(image_copy, cnt.bottommost, cnt.leftmost, color=color, thickness=thickness)
    return image_copy

def get_contour_mask(image, contours, box_color=(0,255,0), box_thickness=3, fill_color=(255,0,255), overlay=None):
    # if overlay is None:
    #     overlay = image
    blank_image = np.zeros(overlay.shape,np.uint8)
    for cnt in contours:
        label, color, confidence = colorclassification.predict(image, cnt.original_contour)
        cv2.drawContours(blank_image, [cnt.original_contour], 0, color, -1)
        # cv2.drawContours(blank_image, [cnt.box], 0, box_color, box_thickness)
    return blank_image

def get_test(image, contours):
    blank_image = np.zeros(image.shape)
    for cnt in contours:
        corners = cnt.get_corners()
        for i in range(len(corners)):
            next_i = (i + 1) % len(corners)
            cv2.line(blank_image, tuple(corners[i]), tuple(corners[next_i]), (0,255,0), 3)
            cv2.circle(blank_image, tuple(corners[i]), 3, (255,255,0), thickness=6)
    return blank_image

# filtering methods
def filterout_byArea(contours,lowerbound=1250,upperbound=35000):
    i = 0
    while i < len(contours): 
        cnt = contours[i]
        if  not (lowerbound < cnt.area < upperbound):
            contours.remove(cnt)
        else:
            i += 1

def filterout_bySolidity(contours,threshold=0.6):
    i = 0
    while i < len(contours): 
        cnt = contours[i]
        if  cnt.solidity < threshold:
            contours.remove(cnt)
        else:
            i += 1

def filterout_byAspectRatio(contours, threshold=0.25):
    i = 0
    while i < len(contours): 
        cnt = contours[i]
        if  abs(cnt.aspect_ratio - 1) > threshold:
            contours.remove(cnt)
        else:
            i += 1

def filterout_byEccentricity(contours, threshold=0.8):
    i = 0
    while i < len(contours): 
        cnt = contours[i]
        if cnt.eccentricity < threshold:
            contours.remove(cnt)
        else:
            i += 1

def filterout_byModeArea(contours, epsilon=3500):
    areas = []
    for cnt in contours:
        areas.append(cnt.rect_area)
    # calculate the floating point mode of the areas
    mode_list = areas
    counts = []
    for a in mode_list:
        for j, (b, c) in enumerate(counts):
            if -epsilon <= a - b <= epsilon:
                counts[j] = (b, c + 1)
                break
        else:
            counts.append((a,1))
    possible_modes = sorted(counts,key=lambda ab: (ab[1],ab[0]))
    possible_modes.reverse()
    print(possible_modes)
    mode = -1
    if len(possible_modes) > 0:
        mode = possible_modes[0][0]

    mode_contours = []
    for j, cnt in enumerate(contours):
        if abs(mode_list[j] - mode) <= epsilon:
            mode_contours.append(cnt)

    while len(contours) > 0:
        contours.remove(contours[0])
    for cnt in mode_contours:
        contours.append(cnt)


# represents some useful information about a given contour
iterator = 0
class Contour:
    def __init__(self, contour):
        global iterator
        # print("Contour: ", iterator)
        self.leftmost = tuple(contour[contour[:,:,0].argmin()][0])
        self.rightmost = tuple(contour[contour[:,:,0].argmax()][0])
        self.topmost = tuple(contour[contour[:,:,1].argmin()][0])
        self.bottommost = tuple(contour[contour[:,:,1].argmax()][0])
        iterator += 1
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
        self.area = cv2.contourArea(contour)
        if self.area == 0:
            return
        self.rect_area = s1 * s2
        self.aspect_ratio = s1/s2
        self.aspect_ratio = 1
        hull = cv2.convexHull(contour)
        self.hull_area = cv2.contourArea(hull)
        self.solidity = float(self.area)/self.hull_area
        if len(self.original_contour) >= 5:
            (x,y),(MA,ma),angle = cv2.fitEllipse(self.original_contour)
            self.center = (int(x),int(y))
            self.axes = (int(MA),int(ma))
            self.angle = angle
            self.eccentricity = MA/ma
        else:
            self.center = (0,0)
            self.axes = (0,0)
            self.angle = 0
            self.eccentricity = 1
    
    # returns a list of points which are calculated to be corners of the contour
    # if the angle to the next point of the contour is greater than the inputted threshold
    def get_corners(self,threshold=10):
        threshold = threshold/180.0 * math.pi
        corners = []
        last = (self.original_contour[1][0][0] - self.original_contour[0][0][0], self.original_contour[1][0][1] - self.original_contour[0][0][1])
        for i in range(1, len(self.original_contour)):
            next_i = (i+6) % len(self.original_contour)
            next = (self.original_contour[next_i][0][0] - self.original_contour[i][0][0], self.original_contour[next_i][0][1] - self.original_contour[i][0][1])
            dot = last[0] * next[0] + last[1] * next[1]
            last_magnitude = math.sqrt(last[0] ** 2 + last[1] ** 2)
            next_magnitude = math.sqrt(next[0] ** 2 + next[1] ** 2)
            angle = math.acos(max(min(dot/(last_magnitude * next_magnitude), 1), -1))
            if angle > threshold:
                corners.append(self.original_contour[i][0])
            last = next
        return corners
        
