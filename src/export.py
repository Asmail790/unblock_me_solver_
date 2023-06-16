from ultralytics import YOLO
import re

model = YOLO(
    '/home/main/Desktop/asmail_kod/drag-and-drop/runs/detect/block_dector_with_grid/weights/best.pt')


model.export(format='torchscript', optimize=True, imgsz=640)
