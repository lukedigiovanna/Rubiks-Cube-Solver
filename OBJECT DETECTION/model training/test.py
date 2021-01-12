from imageai.Detection.Custom import DetectionModelTrainer
import os

executing_cwd = os.getcwd()

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=os.path.join(executing_cwd,"fulldataset"))
trainer.evaluateModel(model_path=os.path.join(executing_cwd,"fulldataset\\models"), 
                      json_path=os.path.join(executing_cwd,"fulldataset\\json\\detection_config.json"), 
                      iou_threshold=0.5, 
                      object_threshold=0.3, 
                      nms_threshold=0.5)
