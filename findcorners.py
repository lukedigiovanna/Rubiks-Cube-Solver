import cv2
import numpy as np
from matplotlib import pyplot as plt
import itertools as it
from shapely.geometry import Polygon


# image = plt.imread("a.jpg")
#plt.imshow(image)

#plt.imshow(image)

missing = (134, 54)
X = np.array([60, 83, 68, 87, 110, 90, 109, 134])
Y = np.array([62, 76, 43, 53, 64, 35,  44, 54])
#X = np.array([31, 48, 68, 37, 53, 72, 42, 57])
#Y = np.array([69, 82, 97, 96, 110, 124, 118, 132])
def find_outlier(corners):
    un = unique_necklaces(corners.tolist())
    #Polygon with maximal area is not crossing itself
    config = un[np.argmax([Polygon(i).area for i in un])]


    min_dist = 100000
    points = [] 

    dist_list = []
    for i in range(len(config)):
        b = list(zip(config[i], config[(i+1) % 4]))

        plt.plot(b[0], b[1], c = "red")
        dist = np.linalg.norm(np.array([b[0][0], b[1][0]]) - np.array([b[0][1],b[1][1]]))

        dist_list.append((dist, (b[0], b[1])))
        if dist < min_dist:
            min_dist = dist
            points = (b[0], b[1])
            #print(dist, points)
    dist_list = sorted(dist_list)

    a = dist_list[0][1]
    b = dist_list[1][1]

    a = list(zip(a[0], a[1]))
    b = list(zip(b[0], b[1]))
    print(a)
    print(b)
    point = [0,0]
    ref0 = []
    ref1 = []
    if a[0] in b:
        point = a[0]
        ref0 = a[1]
        ref1 = b[(b.index(point)+1)%2]
    elif a[1] in b:
        point = a[1]
        ref0 = a[0]
        ref1 = b[(b.index(point)+1)%2]
    corners = corners.tolist()
    other_point = []
    for c in corners:
        if point[0] != c[0] and point[0] != ref0[0] and point[0] != ref1[0]:
            other_point = c
            break
    print([point,ref0,ref1,other_point])
    print(corners)

    return point, dist_list
def unique_necklaces(points):
    L = [points[0], points[1], points[2], points[3]]
    B = it.combinations(L,2)
    swaplist = [e for e in B]
    unique_necklaces = []
    for pair in swaplist:
        necklace = list(L)
        e1 = pair[0]
        e2 = pair[1]
        indexe1 = L.index(e1)
        indexe2 = L.index(e2)
        necklace[indexe1],necklace[indexe2] = necklace[indexe2], necklace[indexe1]
        unique_necklaces.append(necklace)
    return unique_necklaces

def order_points(pts):
  rect = np.zeros((4, 2), dtype = "float32")

  s = pts.sum(axis = 1)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]

  diff = np.diff(pts, axis = 1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]

  if len(set([tuple(i) for i in rect.tolist()])) < 4:
  	rect[0] = pts[np.argmin(pts[:,0])]
  	rect[2] = pts[np.argmax(pts[:,0])]

  	rect[1] = pts[np.argmax(pts[:,1])]
  	rect[3] = pts[np.argmin(pts[:,1])]
  return rect

def four_point_transform (transform, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) **2) + ((br[1] - bl[1]) **2))
    widthB = np.sqrt(((tr[0] - tl[0]) **2) + ((tr[1] - tl[1]) **2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) **2) + ((tr[1] - br[1]) **2))
    heightB = np.sqrt(((tl[0] - bl[0]) **2) + ((tl[1] - bl[1]) **2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
                    [0, 0],
                    [maxWidth -1, 0],
                    [maxWidth -1, maxHeight -1],
                    [0, maxHeight -1]
    ], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective (transform, M, (maxWidth, maxHeight))

    return warped
    # cv2_imshow(fileName)
    # cv2_imshow(warped) 


def get_slope(points):
    return (points[1][1] - points[1][0]) / (points[0][1] - points[0][0])
point_list = np.array(list(zip(X,Y)))
# warped = four_point_transform(image, point_list)


corners = order_points(np.array(list(zip(X,Y))))
print(corners)

outlier = find_outlier(corners)
test = outlier[1]
for i in test:
	print(get_slope(i[1]))
print(outlier[0])

plt.scatter(corners[:,0], corners[:,1], c="blue")
plt.scatter([outlier[0][0]],[outlier[0][1]])
plt.show()