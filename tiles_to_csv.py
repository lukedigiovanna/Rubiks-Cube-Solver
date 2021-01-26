import csv
import os
import cv2

EXECUTION_PATH = os.getcwd()

TILES_FOLDER = os.path.join(EXECUTION_PATH,"tiles")

COLOR = {
    "red": os.path.join(TILES_FOLDER,"red"),
    "green": os.path.join(TILES_FOLDER,"green"),
    "blue": os.path.join(TILES_FOLDER,"blue"),
    "yellow": os.path.join(TILES_FOLDER,"yellow"),
    "orange": os.path.join(TILES_FOLDER,"orange"),
    "white": os.path.join(TILES_FOLDER,"white")
}

csv_file = open("color_ratio_averages.csv", "w", newline='')
writer = csv.writer(csv_file)

# writer.writerow(['blue','green','red','color'])
writer.writerow(['rg','rb','gb'])

for color in COLOR:
    COLOR_FOLDER = os.path.join(TILES_FOLDER,color)
    images = os.listdir(COLOR_FOLDER)
    for image_name in images:
        image = cv2.imread(os.path.join(COLOR_FOLDER,image_name))
        color_average = cv2.mean(image)
        blue = color_average[0]
        green = color_average[1]
        red = color_average[2]
        writer.writerow([red/green,red/blue,green/blue,color])

csv_file.close()