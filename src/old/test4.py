from tools.grid.utils.implementions import DefaultPixelToGridCoordinateTransformer, GridToPixelCoordinateTransformerWithOutGuide,BoundingBox
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results,Boxes
import cv2
from PIL import Image
from PIL.ImageDraw import ImageDraw
import numpy as np
from tools.position.definition import Position
transformerFromPixelToLogical = DefaultPixelToGridCoordinateTransformer()
transformerFromLogicalToPixel =  GridToPixelCoordinateTransformerWithOutGuide()

dector = YOLO("/home/main/Desktop/asmail_kod/drag-and-drop/runs/detect/train/weights/best.pt")

image_path = "/home/main/Desktop/asmail_kod/drag-and-drop/datasets/unblockme/images/train/1696.png"

result:list[Results] = dector.predict(image_path)
boxes:Boxes = result[0].boxes # type: ignore

V3X_ML_CLASS = 3 
GRID_ML_CLASS = 6
index_of_V3X = boxes.cls.tolist().index(V3X_ML_CLASS)
index_of_GRID = boxes.cls.tolist().index(GRID_ML_CLASS)
boundingBoxOfV3X = boxes.xyxy[index_of_V3X]
boundingBoxOfGrid = boxes.xyxy[index_of_GRID]

x_start_grid,x_end_grid = (boundingBoxOfGrid[[0,2]]).tolist()
y_start_grid,y_end_grid = (boundingBoxOfGrid[[1,3]]).tolist()

x_start_block,y_start_block = boundingBoxOfV3X[0:2].tolist()




logical_pos = transformerFromPixelToLogical(
    gridImageProperties=BoundingBox(x_start_grid,x_end_grid,y_start_grid,y_end_grid ),
    coordinate=(x_start_block,y_start_block),
    )
img = Image.open(image_path)

drawer = ImageDraw(img)
drawer.rectangle(((x_start_block,y_start_block),(x_start_block+boxes.xywh[index_of_V3X][2],y_start_block+ boxes.xywh[index_of_V3X][3]) ) )

print(logical_pos)
cv2.imshow('img',np.array(img) )
cv2.waitKey(3000)

for new_logical_pos in [Position(x,y) for x in range(6) for y in range(6)]:
    pixel_pos_x,pixel_pos_y = transformerFromLogicalToPixel(
        position=new_logical_pos,
        gridImageProperties=BoundingBox(x_start_grid,x_end_grid,y_start_grid,y_end_grid ),
    )

    #print((pixel_pos_x,pixel_pos_y))
    #print((x_start_block,y_start_block))

    img = Image.open(image_path)

    zz:tuple[float,float] = (pixel_pos_x + boxes.xywh[index_of_V3X][2], pixel_pos_y +  boxes.xywh[index_of_V3X][3])  # type: ignore

    drawer = ImageDraw(img)
    drawer.rectangle(((pixel_pos_x,pixel_pos_y),zz ) )

    img.show()
    cv2.imshow('img',np.array(img) )
    cv2.waitKey(500)
    img.close()

    #print(boxes.numpy)

    
