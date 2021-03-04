
import os
import keras
import cv2
import numpy as np

EXECUTION_DIRECTORY = os.getcwd()

CONFIDENCE_THRESHOLD = 0.0

# load color classification model
color_model = keras.models.load_model(os.path.join(EXECUTION_DIRECTORY,"colorclassification.h5"))
color_ratio_model = keras.models.load_model(os.path.join(EXECUTION_DIRECTORY,"colorratioclassification.h5"))

# stores the class labels respective to the model prediction output
color_labels = ('blue','green','orange','red','white','yellow')
color_values = ((255,0,0),(0,255,0),(0,125,255),(0,0,255),(255,255,255),(0,255,255))

# helper function to return the maximum index in the given array
def max_index(inlist):
    ind = 0
    max = inlist[ind]
    for i in range(1,len(inlist)):
        if inlist[i] > max:
            max = inlist[i]
            ind = i
    return ind
# helper function to determine whether a prediction can be trusted
def has_confidence(inlist):
    for val in inlist:
        if val > CONFIDENCE_THRESHOLD:
            return True
    return False
# predicts the color label of a mean color using raw rgb color model
def predict_using_raw_rgb(mean):
    predicted = color_model.predict([[mean[0]/255,mean[1]/255,mean[2]/255]]).tolist()
    max_ind = max_index(predicted[0])
    color_label = color_labels[max_ind]
    color = color_values[max_ind]
    confidence = predicted[0][max_ind]
    return color_label, color, confidence

# predicts the color label of a mean color using ratios of red green and blue color model
# this model is recommended over the RGB model as it is more accurate
def predict_using_ratios(mean):
    rg = mean[2]/mean[1]
    rb = mean[2]/mean[0]
    gb = mean[1]/mean[0]
    predicted = color_ratio_model.predict([[rg,rb,gb]]).tolist()
    max_ind = max_index(predicted[0])
    color_label = color_labels[max_ind]
    color = color_values[max_ind]
    confidence = predicted[0][max_ind]
    return color_label, color, confidence

def predict(image, contour):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(image_gray.shape,np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    mean_val = cv2.mean(image,mask=mask)
    return predict_using_ratios(mean_val)