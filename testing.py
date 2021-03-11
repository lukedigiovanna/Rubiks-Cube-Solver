from rubiks_regions import *
from rubik_solver import utils

dp = "all rubiks images/rubiks_corners_3/"

one = Image(dp+"0.JPG")
print("Finished image one")
cv2.imwrite(dp+"output/0image.png",one.image)
cv2.imwrite(dp+"output/0threshold.png",one.threshold)
cv2.imwrite(dp+"output/0mask.png",one.mask)

save_face_image(dp+"output/0topface.png",one.top_face)
save_face_image(dp+"output/0leftface.png",one.left_face)
save_face_image(dp+"output/0rightface.png",one.right_face)

two = Image(dp+"1.JPG")
print("Finished image two")
cv2.imwrite(dp+"output/1image.png",two.image)
cv2.imwrite(dp+"output/1threshold.png",two.threshold)
cv2.imwrite(dp+"output/1mask.png",two.mask)

save_face_image(dp+"output/1topface.png",two.top_face)
save_face_image(dp+"output/1leftface.png",two.left_face)
save_face_image(dp+"output/1rightface.png",two.right_face)

print("Top one")
print(one.top_face)
print("Left one")
print(one.left_face)
print("Right one")
print(one.right_face)
print("Left two")
print(two.left_face)
print("Right two")
print(two.right_face)

rubiks = ["u"]*54
for i in range(9):
    rubiks[i + 18] = one.top_face[int(i/3)][i % 3][0]
    rubiks[i + 45] = one.left_face[int(i/3)][i % 3][0]
    rubiks[i + 27] = one.right_face[i % 3][2-int(i/3)][0]
    rubiks[i] = two.left_face[2-int(i/3)][2 - i % 3][0]
    rubiks[i + 9] = two.right_face[2 - i % 3][int(i/3)][0]
rubiks[36] = 'b'
rubiks[37] = 'w'
rubiks[38] = 'b'
rubiks[39] = 'o'
rubiks[40] = 'w'
rubiks[41] = 'y'
rubiks[42] = 'y'
rubiks[43] = 'y'
rubiks[44] = 'g'
rubiks = "".join(rubiks)
print(rubiks)
# print("solution: ")
# print(utils.solve(rubiks,'Beginner'))
# centers = []
# for i in range(6):
#     centers.append(rubiks[4 + 9 * i])
# colors = ['W','O','B','Y','G','R']
# last_center = 'U'
# for c in colors:
#     if c not in centers:
#         last_center = c
#         break
# print(centers)
# print(last_center) 
cube_image = np.zeros((100 * 9, 100 * 12, 3),np.uint8)
colors = {
    'r': (0,0,255),
    'g': (0,255,0),
    'b': (255,0,0),
    'y': (0,255,255),
    'o': (0,125,255),
    'w': (255,255,255)
}
def color_tile(x, y, letter):
    cv2.rectangle(cube_image, (x * 100, y * 100), (x * 100 + 100, y * 100 + 100), colors[letter], -1)
    cv2.rectangle(cube_image, (x * 100, y * 100), (x * 100 + 100, y * 100 + 100), (0,0,0), 5)
for i in range(9):
    color_tile(3 + i % 3, int(i/3), rubiks[i])
    color_tile(i % 3, 3 + int(i/3), rubiks[i+9])
    color_tile(3 + i % 3, 3 + int(i/3), rubiks[i+18])
    color_tile(6 + i % 3, 3 + int(i/3), rubiks[i+27])
    color_tile(9 + i % 3, 3 + int(i/3), rubiks[i+36])
    color_tile(3 + i % 3, 6 + int(i/3), rubiks[i+45])
cv2.imwrite(dp+"output/fullcube.png",cube_image)