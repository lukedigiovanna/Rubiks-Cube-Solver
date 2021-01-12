# splits a dataset into train/validation data

import os
import shutil
import random

tvr = 0.75

executing_location = os.getcwd()
dataset_folder_name = "fulldataset"

dataset_folder = os.path.join(executing_location,dataset_folder_name)

images_folder = os.path.join(dataset_folder,"images")
annotations_folder = os.path.join(dataset_folder,"annotations")

images = []
annotations = []

outDirectory = os.path.join(dataset_folder,"test")

num_images = 0
while os.path.exists(os.path.join(images_folder,str(num_images)+".jpg")):
    num_images+=1

train_indices = []
num_of_train = int(tvr * num_images)
while len(train_indices) < num_of_train:
    i = random.randint(0,num_images-1)
    if i in train_indices:
        continue
    else:
        train_indices.append(i)
validation_indices = []
while len(train_indices) + len(validation_indices) < num_images:
    i = random.randint(0,num_images-1)
    if i in train_indices or i in validation_indices:
        continue
    else:
        validation_indices.append(i)

print(train_indices)
print(validation_indices)

train_folder = os.path.join(dataset_folder,"train")
train_images_folder = os.path.join(train_folder,"images")
train_annotations_folder = os.path.join(train_folder,"annotations")
validation_folder = os.path.join(dataset_folder,"validation")
validation_images_folder = os.path.join(validation_folder,"images")
validation_annotations_folder = os.path.join(validation_folder,"annotations")

for index in train_indices:
    shutil.copy2(os.path.join(images_folder,str(index)+".jpg"),train_images_folder)
    shutil.copy2(os.path.join(annotations_folder,str(index)+".xml"),train_annotations_folder)

for index in validation_indices:
    shutil.copy2(os.path.join(images_folder,str(index)+".jpg"),validation_images_folder)
    shutil.copy2(os.path.join(annotations_folder,str(index)+".xml"),validation_annotations_folder)

print("done")   