from ultralytics import YOLO
import re

model = YOLO('yolov8n.pt')
regex = r"model\.[0-9]\..+"
for k, v in model.model.named_parameters():
    if re.match(regex, k):
        v.requires_grad = False


model.train(data='dataset_info.yaml', epochs=10, imgsz=360)
