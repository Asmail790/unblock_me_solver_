from tools.grid.utils.implementions import DefaultPixelToGridCoordinateTransformer, GridToPixelCoordinateTransformerWithOutGuide,BoundingBox,ConvertImageToGrid,BoundingBox
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results,Boxes
import cv2
from PIL import Image
from PIL.ImageDraw import ImageDraw
import numpy as np
from tools.position.definition import Position
from tools.grid.definition import Grid
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



x_start_block,x_end_block = (boundingBoxOfV3X[[0,2]]).tolist()
y_start_block,y_end_block = (boundingBoxOfV3X[[1,3]]).tolist()
from attrs import evolve

gridprops = BoundingBox(x_start_grid,x_end_grid,y_start_grid,y_end_grid )
blockprops = [BoundingBox(V3X_ML_CLASS,x_start_block,x_end_block,y_start_block,y_end_block )]
helper = ConvertImageToGrid(gridInfo=gridprops,blockproperties=blockprops)

for new_logical_pos in [Position(x,y) for x in range(6) for y in range(6)]:
    img = Image.open(image_path)
    grid = helper.to_grid()
    props = helper.image_position_for_block(list(grid.blocks)[0], new_position = new_logical_pos)
   


    drawer = ImageDraw(img)
    drawer.rectangle( ((props.left,props.top),(props.right,props.bottom)) )

    cv2.imshow('img',np.array(img) )
    cv2.waitKey(500)
    img.close()

    #print(boxes.numpy)

    
