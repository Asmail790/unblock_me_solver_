from ultralytics import YOLO
import re

model = YOLO('yolov8s.pt')
regex = r"model\.[0-9]\..+"
for k, v in model.model.named_parameters():
    if re.match(regex, k):
        v.requires_grad = False


model.train(
    data='new_unblock_me_with_grid.yaml',
    batch=32,
    epochs=15,
    imgsz=640,
    cache=True,
    workers=4,
    name='block_dector_with_grid'
)
