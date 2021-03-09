"""

Objective of this attempt:

1. Threshold the image
2. Find regions that are completely enclosed in white regions

"""

import cv2
import numpy as np
import random
import colorclassification
import json
import requests
import urllib

image = cv2.imread("all rubiks images/rubiks_corners_no_background/0.png")
image = cv2.imread("all rubiks images/rubiks_corners/0.JPG")

def remove_background(image):
    cv2.imwrite("temp.jpg", image)
    params = (
        ('api_key', 'ak-shiny-lab-a046e44'),
    )

    files = {
        'file': ('temp.jpg', open('temp.jpg', 'rb')),
    }
    response = requests.post('https://api.boring-images.ml/v1.0/transparent-net', params=params, files=files)
    url = json.loads(response.text)["result"]

    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

image = remove_background(image)

target_width = 180

aspect_ratio = image.shape[1]/image.shape[0] # w/h
dim = (int(target_width),int(target_width/aspect_ratio))
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
gray_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2GRAY)

threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 2)
dilation_kernel = np.ones((3,3))
threshold = 255-cv2.dilate(threshold, dilation_kernel)

cv2.imwrite("threshold.png",threshold)

cache = []

row_i = 0
height, width = threshold.shape

index = 0
class Region:
    def __init__(self, pixel_coordinates):
        global index
        self.pixel_list = np.array(pixel_coordinates)
        if len(pixel_coordinates) < 75 or len(pixel_coordinates) > 500:
            return None
        self.leftx = self.pixel_list[:,0][self.pixel_list[:,0].argmin()]
        self.rightx = self.pixel_list[:,0][self.pixel_list[:,0].argmax()]
        self.topy = self.pixel_list[:,1][self.pixel_list[:,1].argmin()]
        self.bottomy = self.pixel_list[:,1][self.pixel_list[:,1].argmax()]
        map_width = self.rightx - self.leftx + 1
        map_height = self.bottomy - self.topy + 1
        self.map = np.zeros((map_height,map_width), np.uint8)
        for i in range(self.topy,self.bottomy+map_height):
            for j in range(self.leftx,self.rightx+map_width):
                if (j,i) in pixel_coordinates:
                    self.map[i-self.topy][j-self.leftx] = 255
        self.edges = np.zeros(self.map.shape, np.uint8)
        self.contours = []
        # make the edge map and check for convexity
        for i in range(0, map_height):
            for j in range(0, map_width):
                if i == 0 or i == map_height - 1 or j == 0 or j == map_width - 1:
                    self.edges[i][j] = self.map[i][j]
                    if self.edges[i][j] == 255:
                        self.contours.append((j+self.leftx, i+self.topy))
                    continue
                if self.map[i][j] == 255 and (self.map[i-1][j] == 0 or self.map[i+1][j] == 0 or self.map[i][j-1] == 0 or self.map[i][j+1] == 0):
                    self.edges[i][j] = 255
                    self.contours.append((j+self.leftx,i+self.topy))
        self.left_edge = (0, 0)
        for i in range(0, map_height):
            if self.map[i][0] == 255:
                self.left_edge = (0, i)
                break
        self.right_edge = (0, 0)
        for i in range(0, map_height):
            if self.map[i][map_width-1] == 255:
                self.right_edge = (map_width-1, i)
                break
        self.slope = (self.right_edge[1] - self.left_edge[1])/(self.right_edge[0] - self.left_edge[0])
        

        # convex detection
        self.is_convex = True
        # for i in range(0, map_height):
        #     edge_count = 0
        #     if self.map[i][0] == 255:
        #         edge_count+=1
        #     for j in range(0, map_width-1):
        #         if self.map[i][j] == 0 and self.map[i][j+1] == 255:
        #             edge_count += 1
        #             if edge_count > 1:
        #                 self.is_convex = False
        #                 break
        #     if not self.is_convex:
        #         break # no need to keep checking if we already found out we are not convex
        # if self.is_convex:
        #     for j in range(0, map_width):
        #         edge_count = 0
        #         if self.map[0][j] == 255:
        #             edge_count += 1
        #         for i in range(0, map_height-1):
        #             if self.map[i][j] == 0 and self.map[i+1][j] == 255:
        #                 edge_count += 1
        #                 if edge_count > 1:
        #                     self.is_convex = False
        #                     break
        #         if not self.is_convex:
        #             break

        cv2.imwrite("test/region"+str(index)+".png",self.map)
        cv2.imwrite("test/edges"+str(index)+".png",self.edges)
        index+=1

    def get_predicted_color(self):
        return predict_color(self.pixel_list)

    def get_area(self):
        return len(self.pixel_list)

    def is_touching_edge(self, image):
        return self.leftx == 0 or self.rightx == image.shape[1] - 1 or self.topy == 0 or self.bottomy == image.shape[0] - 1


# returns a list of the pixel coordinates that make up the region
def find_region(x, y, region, count=0):
    point = (x, y)
    if count >= 500 or x < 0 or x >= width or y < 0 or y >= height or point in cache or threshold[y][x] == 0:
        return []
    else:
        region.append(point)
        cache.append(point)
        find_region(x + 1, y, region, count + 1)
        find_region(x - 1, y, region, count + 1)
        find_region(x, y + 1, region, count + 1)
        find_region(x, y - 1, region, count + 1)

def average_color(region):
    mean = [0,0,0]
    for pixel in region:
        color = image[pixel[1]][pixel[0]]
        for i in range(3):
            mean[i] += color[i]
    for i in range(3):
        mean[i] /= len(region)
    return mean

def predict_color(region):
    mean = average_color(region)
    label, color, confidence = colorclassification.predict_using_ratios(mean)
    if confidence < 0.5:
        return (255,0,255)
    else:
        return color

blank = np.zeros(image.shape, np.uint8)

while row_i < height:
    col_i = 0
    while col_i < width:
        region = []
        find_region(col_i, row_i, region)
        region = Region(region)
        if len(region.pixel_list) < 75 or len(region.pixel_list) > 500 or region.is_touching_edge(image) or not region.is_convex:
            col_i += 1
            continue
        color = region.get_predicted_color()
        # color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        outline_color = (255,0,0)
        if region.slope < -0.25:
            outline_color = (0,255,0)
        elif region.slope > 0.25:
            outline_color = (0,0,255)
        for p in region.pixel_list:
            blank[p[1]][p[0]] = color
        cv2.rectangle(blank,(region.leftx,region.topy),(region.rightx,region.bottomy),outline_color,thickness=1)
        for cnt_point in region.contours:
            cv2.circle(image, cnt_point, 0, (0,255,0), thickness=1)
        col_i += 1
    row_i += 1

cv2.imwrite("regions.png", blank)
cv2.imwrite("contours.png", image)