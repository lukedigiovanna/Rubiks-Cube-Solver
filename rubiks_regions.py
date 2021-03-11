
import cv2
import colorclassification
import numpy as np
import json
import requests
import urllib

target_width = 180

def remove_background(filepath):
    params = (
        ('api_key', 'ak-shiny-lab-a046e44'),
    )

    files = {
        'file': (filepath, open(filepath, 'rb')),
    }
    response = requests.post('https://api.boring-images.ml/v1.0/transparent-net', params=params, files=files)
    url = json.loads(response.text)["result"]

    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

class Image:
    def __init__(self, filepath):
        self.image = remove_background(filepath)
        aspect_ratio = self.image.shape[1]/self.image.shape[0] # w/h
        dim = (int(target_width),int(target_width/aspect_ratio))
        self.image = cv2.resize(self.image, dim, interpolation = cv2.INTER_AREA)

        # colorscale the image for filtering
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        gray_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2GRAY)

        # filter the image
        self.threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
        dilation_kernel = np.ones((3,3))
        self.threshold = 255-cv2.dilate(self.threshold, dilation_kernel)

        self.height, self.width = self.threshold.shape

        # remove the white region around the cube
        for i in range(dim[1]):
            for j in range(dim[0]):
                if self.threshold[i][j] == 0:
                    break
                else:
                    self.threshold[i][j] = 0
            for j in range(dim[0]-1, -1, -1):
                if self.threshold[i][j] == 0:
                    break
                else:
                    self.threshold[i][j] = 0
        
        # find regions
        self.regions = []

        self.top_face_regions = []
        self.left_face_regions = []
        self.right_face_regions = []

        self.cache = []

        row_i = 0
        while row_i < self.height:
            col_i = 0
            while col_i < self.width:
                region = []
                self.find_region(col_i, row_i, region)
                region = Region(region)
                if len(region.pixel_list) < 50 or len(region.pixel_list) > 500:
                    col_i += 1
                    continue
                self.regions.append(region)
                if region.total_grade <= -5:
                    self.right_face_regions.append(region)
                elif region.total_grade >= 5:
                    self.left_face_regions.append(region)
                else:
                    self.top_face_regions.append(region)
                col_i += 1
            row_i += 1



    # returns a list of the pixel coordinates that make up the region
    def find_region(self, x, y, region, count=0):
        point = (x, y)
        if count >= 500 or x < 0 or x >= self.width or y < 0 or y >= self.height or point in self.cache or self.threshold[y][x] == 0:
            return []
        else:
            region.append(point)
            self.cache.append(point)
            self.find_region(x + 1, y, region, count + 1)
            self.find_region(x - 1, y, region, count + 1)
            self.find_region(x, y + 1, region, count + 1)
            self.find_region(x, y - 1, region, count + 1)

    def average_color(self, region):
        mean = [0,0,0]
        for pixel in region:
            color = self.image[pixel[1]][pixel[0]]
            for i in range(3):
                mean[i] += color[i]
        for i in range(3):
            mean[i] /= len(region)
        return mean

        
class Region:
    def __init__(self, image, pixel_coordinates):
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

        mean = image.average_color(self.pixel_list)
        self.color_label, self.color, confidence = colorclassification.predict_using_ratios(mean)

    def get_predicted_color(self):
        return self.color

    def get_area(self):
        return len(self.pixel_list)

    def is_touching_edge(self, image):
        return self.leftx == 0 or self.rightx == image.shape[1] - 1 or self.topy == 0 or self.bottomy == image.shape[0] - 1

