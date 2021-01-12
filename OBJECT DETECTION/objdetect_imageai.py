from imageai.Detection.Custom import CustomObjectDetection
import os

execution_path = os.getcwd()

# load the model
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path,"OBJECT DETECTIOn/detect_cubes_loss_11.h5"))
detector.setJsonPath(os.path.join(execution_path,"OBJECT DETECTION/detection_config.json"))
detector.loadModel()

detections, extracted_images = detector.detectObjectsFromImage(
    input_image=os.path.join(execution_path,"OBJECT DETECTION/test_images/135.jpg"), 
    output_image_path=os.path.join(execution_path , "OBJECT_DETECTION/imagenew.jpg"), 
    extract_detected_objects=True)

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"])

if len(detections) == 0:
    print("No Rubik's Cube Found")

for i in range(200):
    detections, extracted_images = detector.detectObjectsFromImage(
        input_image=os.path.join(execution_path,"OBJECT DETECTION/test_images/rubiks/"+str(i)+".jpg"),
        output_image_path=os.path.join(execution_path,"OBJECT DETECTION/test_images/rubiks_out/"+str(i)+"out.jpg"),
        extract_detected_objects=True
    )

    for eachObject in detections:
        print(eachObject["name"], " : ", eachObject["percentage_probability"])