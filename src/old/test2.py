import pathlib
from tools.grid.utils.implementions import SimpleDataSetGenerator,ToYOLOV5
import logging
import os,pickle
from tools.block.definition import VerticalMovableBlock2X,VerticalMovableBlock3X,HorizontalMovableBlock2X,HorizontalMovableBlock3X,FixedBlock,MainBlock
from tools.grid.definition import Grid




source = pathlib.Path("/home/main/Desktop/asmail_kod/drag-and-drop/datasets/simple_data_set")


#SimpleDataSetGenerator(source).generate()
destination_image_folder = pathlib.Path("/home/main/Desktop/asmail_kod/drag-and-drop/datasets/new_yolo_dataset/images")
destination_labels_folder = pathlib.Path("/home/main/Desktop/asmail_kod/drag-and-drop/datasets/new_yolo_dataset/labels")
themes_source = pathlib.Path("/home/main/Desktop/asmail_kod/drag-and-drop/resources/ORIGINAL_THEME_1") 
guide_image_path = pathlib.Path("/home/main/Desktop/asmail_kod/drag-and-drop/resources/position_guide.png")

logging.basicConfig(filename=source.parent / "logs.txt" , level=logging.INFO)

converter = ToYOLOV5(source,destination_labels_folder,destination_image_folder, themes_source,guide_image_path)
converter.generate()

def stats():

    length_count = {i:0 for i in range(20)}

    fixedBlock_count = 0
    H2X_count = 0
    H3X_count = 0
    V2X_count= 0
    V3X_count = 0
    MainBlock_count =  0
    filnenames = os.listdir(source)
    total_nbr_of_grids = len(filnenames)
    for filename in filnenames:
        grid:Grid = pickle.load(open(source / filename, "rb"))
        length_count[len(grid.blocks)]+=1
        for block in grid.blocks:
            if isinstance(block,MainBlock):
                MainBlock_count+=1
            
            elif isinstance(block,FixedBlock):
                fixedBlock_count+=1
            
            elif isinstance(block,VerticalMovableBlock2X):
                V2X_count +=1
            
            elif isinstance(block,VerticalMovableBlock3X):
                V3X_count +=1
            
            elif isinstance(block,HorizontalMovableBlock2X):
                H2X_count +=1
            
            elif isinstance(block,HorizontalMovableBlock3X ):
                H3X_count +=1

            else:
                print(type(block))
                raise ValueError()

    total = fixedBlock_count + H3X_count + H2X_count + V2X_count + V3X_count + MainBlock_count 

    fixedBlock_procent = fixedBlock_count/total
    MainBlock_procent = MainBlock_count/total 


    H2X_procent = H2X_count/total
    H3X_procent = H3X_count/total

    V2X_procent = V2X_count/total
    V3X_procent = V3X_count/total


    print(
    f"""
    procent statics

    fixedBlock_procent = {fixedBlock_procent}
    MainBlock_procent = {MainBlock_procent}
    H2X_procent = {H2X_procent}
    H3X_procent = {H3X_procent}
    V2X_procent = {V2X_procent}
    V3X_procent = {V3X_procent}

    count statics
    fixedBlock_procent = {fixedBlock_count}
    MainBlock_count = {MainBlock_count}
    H2X_count = {H2X_count}
    H3X_count = {H3X_count}
    V2X_count= {V2X_count}
    V3X_count = {V3X_count}

    -------------------------

    length_stats = {({ k:round(v/total_nbr_of_grids,4) for k,v in length_count.items()})}
    """
    )


