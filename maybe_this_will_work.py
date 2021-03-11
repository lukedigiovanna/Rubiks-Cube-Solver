"""

Objective of this attempt:

1. Threshold the image
2. Find regions that are completely enclosed in white regions

"""

import cv2
import numpy as np
import colorclassification
import json
import requests
import urllib

image = cv2.imread("all rubiks images/rubiks_corners_no_background/0.png")
image = cv2.imread("all rubiks images/rubiks_corners_3/0.JPG")

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

# for i in range(3,51,2):
threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
dilation_kernel = np.ones((3,3))
threshold = 255-cv2.dilate(threshold, dilation_kernel)
    # cv2.imwrite("threshold"+str(i)+".png",threshold)
cv2.imwrite("yuv.png", cv2.cvtColor(image, cv2.COLOR_BGR2YUV))

for i in range(dim[1]):
    for j in range(dim[0]):
        if threshold[i][j] == 0:
            break
        else:
            threshold[i][j] = 0
    for j in range(dim[0]-1, -1, -1):
        if threshold[i][j] == 0:
            break
        else:
            threshold[i][j] = 0

cv2.imwrite("threshold.png",threshold)

cache = []

row_i = 0
height, width = threshold.shape

index = 0
class Region:
    def __init__(self, pixel_coordinates):
        global index
        self.pixel_list = np.array(pixel_coordinates)
        if len(pixel_coordinates) < 50 or len(pixel_coordinates) > 500:
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
        self.center = (self.leftx + int(map_width/2), self.topy + int(map_height/2))
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

        ys = []
        for j in range(map_width):
            for i in range(map_height):
                if self.map[i][j] == 255:
                    ys.append(i)
                    break
        self.total_grade = 0
        for i in range(map_width-1):
            if ys[i] < ys[i+1]:
                self.total_grade += 1
            elif ys[i] > ys[i+1]:
                self.total_grade -= 1

        mean = average_color(self.pixel_list)
        self.color_label, self.color, confidence = colorclassification.predict_using_ratios(mean)

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

top_face_regions = []
left_face_regions = []
right_face_regions = []

while row_i < height:
    col_i = 0
    while col_i < width:
        region = []
        find_region(col_i, row_i, region)
        region = Region(region)
        if len(region.pixel_list) < 50 or len(region.pixel_list) > 500:
            col_i += 1
            continue
        
        color = region.get_predicted_color()

        outline_color = (255,0,0)
        if region.total_grade <= -5:
            outline_color = (0,255,0)
            right_face_regions.append(region)
        elif region.total_grade >= 5:
            outline_color = (0,0,255)
            left_face_regions.append(region)
        else:
            top_face_regions.append(region)

        for p in region.pixel_list:
            blank[p[1]][p[0]] = color
        cv2.rectangle(blank,(region.leftx,region.topy),(region.rightx,region.bottomy),outline_color,thickness=1)
        
        cv2.circle(blank, region.center, 1, (255,255,255), thickness=2)
        
        col_i += 1
    row_i += 1


cv2.imwrite("regions.png",blank)


# take faces and put identified sides into simple data structure
top_face = [['white','white','white'],['white','white','white'],['white','white','white']]
left_face = [['white','white','white'],['white','white','white'],['white','white','white']]
right_face = [['white','white','white'],['white','white','white'],['white','white','white']]

def print_face(face):
    for row in face:
        print(row)

color_dict = {
    "red": (0,0,255),
    "orange": (0,125,255),
    "white": (255, 255, 255),
    "yellow": (0,255,255),
    "blue":(255,0,0),
    "green":(0,255,0)
}
def save_face_image(image_name, face_labels):
    output = np.zeros([300,300,3],np.uint8)
    for i in range(3):
        for j in range(3):
            cv2.rectangle(output, (j * 100, i * 100), (j * 100 + 100, i * 100 + 100), color_dict[face_labels[i][j]], -1)
            cv2.rectangle(output, (j * 100, i * 100), (j * 100 + 100, i * 100 + 100), (0,0,0), 5)
    cv2.imwrite(image_name+".png", output)

top_face_regions.sort(key=lambda a: a.center[1])
top_face[0][0] = top_face_regions[0].color_label
second_row = [top_face_regions[1], top_face_regions[2]]
second_row.sort(key=lambda a: a.center[0])
top_face[0][1] = second_row[1].color_label
top_face[1][0] = second_row[0].color_label
third_row = [top_face_regions[3],top_face_regions[4],top_face_regions[5]]
third_row.sort(key=lambda a: a.center[0])
top_face[2][0] = third_row[0].color_label
top_face[1][1] = third_row[1].color_label
top_face[0][2] = third_row[2].color_label
fourth_row = [top_face_regions[6],top_face_regions[7]]
fourth_row.sort(key=lambda a: a.center[0])
top_face[2][1] = fourth_row[0].color_label
top_face[1][2] = fourth_row[1].color_label
top_face[2][2] = top_face_regions[8].color_label
print_face(top_face)

left_face_regions.sort(key=lambda a: a.center[0])
first_col = [left_face_regions[0],left_face_regions[1],left_face_regions[2]]
first_col.sort(key=lambda a: a.center[1])
second_col = [left_face_regions[3],left_face_regions[4],left_face_regions[5]]
second_col.sort(key=lambda a: a.center[1])
third_col = [left_face_regions[6],left_face_regions[7],left_face_regions[8]]
third_col.sort(key=lambda a: a.center[1])
for i in range(3):
    left_face[i][0] = first_col[i].color_label
    left_face[i][1] = second_col[i].color_label
    left_face[i][2] = third_col[i].color_label
print_face(left_face)

right_face_regions.sort(key=lambda a: a.center[0])
first_col = [right_face_regions[0],right_face_regions[1],right_face_regions[2]]
first_col.sort(key=lambda a: a.center[1])
second_col = [right_face_regions[3],right_face_regions[4],right_face_regions[5]]
second_col.sort(key=lambda a: a.center[1])
third_col = [right_face_regions[6],right_face_regions[7],right_face_regions[8]]
third_col.sort(key=lambda a: a.center[1])
for i in range(3):
    right_face[i][0] = first_col[i].color_label
    right_face[i][1] = second_col[i].color_label
    right_face[i][2] = third_col[i].color_label
print_face(right_face)

save_face_image("top", top_face)
save_face_image("left", left_face)
save_face_image("right", right_face)