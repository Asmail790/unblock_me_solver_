from pathlib import Path
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results
from tools.repsentantions.asObjects.utils.guiders.default import DeafaultJPI
from tools.common.interafaces.java_to_python_interface import JavaToPythonInterFace, BoundingBox
from tools.common.constants import ML_ClASS_NBR, is_ML_NBR
from tools.repsentantions.asObjects.utils.datagenerators.yolov5datasetgenerator import AddGridLabel
model = YOLO(
    '/home/main/Desktop/asmail_kod/drag-and-drop/runs/detect/train/weights/best.pt')
# x1,y2 = 10,203
# x2,y2 = 283,477
# width = 295,640

AddGridLabel(
    Path("/home/main/Desktop/asmail_kod/drag-and-drop/datasets/new_yolo_dataset/labels"),
    Path("/home/main/Desktop/asmail_kod/drag-and-drop/datasets/with_grid_label/labels"),
    10, 283, 203, 477, 295, 640
)
exit()


def test(src: Path):
    interface: JavaToPythonInterFace = DeafaultJPI()

    bounding_boxes = []
    result: list[Results] = model.predict(src)
    boxes = result[0].boxes
    s = boxes.shape
    for (x1, y1, x2, y2), c in zip(boxes.xyxy.tolist(), boxes.cls.tolist()):
        clazz = int(c)
        if is_ML_NBR(clazz):
            bounding_boxes.append(BoundingBox(clazz, x1, x2, y1, y2))

    print(interface.guide(bounding_boxes))


test(Path("/home/main/Desktop/asmail_kod/drag-and-drop/datasets/unblockme/images/train/6.png"))
