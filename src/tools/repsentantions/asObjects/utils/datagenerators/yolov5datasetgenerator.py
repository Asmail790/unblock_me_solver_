import numpy as np
from PIL.ImageDraw import ImageDraw
from cv2 import imshow, waitKey
import pickle

from dataclasses import dataclass
from PIL.Image import Image as PILIMAGE
from PIL import Image
from os import listdir
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
from tools.repsentantions.asObjects.definitions.grid import Grid
from tools.repsentantions.asObjects.utils.converters.coordinate import DefaultGridToPixelCoordinateConverterWithGuide
from tools.repsentantions.asObjects.definitions.block import Block
from tools.repsentantions.asObjects.definitions.block import H2XBlock, H3XBlock, V2XBlock, V3XBlock, MainBlock, FixedBlock
from tools.common.constants import ML_STR_TO_ML_NBR
from tools.repsentantions.asObjects.utils.converters.block import PY_BLOCK_OBJECT_TO_ML_STR


def to_ML_nbr(x: type[Block]):
    name = PY_BLOCK_OBJECT_TO_ML_STR[x]
    nbr = ML_STR_TO_ML_NBR[name]
    return nbr


@dataclass(frozen=True)
class YoloLabel():
    clazz: int
    x_center: float
    y_center: float
    w: float
    h: float

    def __str__(self):
        return "{} {:.5f} {:.5f} {:.5f} {:.5f}".format(
            self.clazz, self.x_center, self.y_center, self.w, self.h)


@dataclass(frozen=True)
class YoloSample:
    labels: list[YoloLabel]
    image: PILIMAGE

    def __str__(self) -> str:

        return "\n".join([str(x) for x in self.labels])


class ToYOLOV5():
    def __init__(self,
                 source_samples_folder: Path,
                 destination_labels_folder: Path,
                 destination_image_folder: Path,
                 theme_blocks_folder_path: Path,
                 guide_position_img_path: Path

                 ):
        self.__source_samples_folder = source_samples_folder
        self.__coordinateTransformer = DefaultGridToPixelCoordinateConverterWithGuide(
            guide_position_img_path)

        self.__destination_image_folder = destination_image_folder
        self.__destination_labels_folder = destination_labels_folder
        self.__background_image = Image.open(
            theme_blocks_folder_path / "background.png")

        _3XV_img = Image.open(theme_blocks_folder_path / "3XV.png")
        _2XV_img = Image.open(theme_blocks_folder_path / "2XV.png")
        _3XH_img = Image.open(theme_blocks_folder_path / "3XH.png")
        _2XH_img = Image.open(theme_blocks_folder_path / "2XH.png")
        _MainBlock_img = Image.open(theme_blocks_folder_path / "MainBlock.png")
        _FixedBlock_img = Image.open(
            theme_blocks_folder_path / "FixedBlock.png")

        self.__block_class_to_img: dict[type[Block], PILIMAGE] = {
            MainBlock: _MainBlock_img,
            FixedBlock: _FixedBlock_img,
            H2XBlock: _2XH_img,
            H3XBlock: _3XH_img,
            V2XBlock: _2XV_img,
            V3XBlock: _3XV_img
        }

    def __generate_sample(self, grid: Grid) -> YoloSample:
        block_labels = list()

        width, height = self.__background_image.size
        new_hight = 640
        aspect = new_hight / height

        new_width = aspect * width
        new_size = int(new_width), int(new_hight)
        grid_image = self.__background_image.copy()

        for block in grid:
            x, y = self.__coordinateTransformer(block.pos)
            block_img = self.__block_class_to_img[block.__class__]
            block_width, block_height = self.__block_class_to_img[block.__class__].size

            block_position_in_image = (int(x), int(y))
            grid_image.paste(block_img, block_position_in_image, block_img)

            center_xn = (x + block_width / 2) / \
                self.__coordinateTransformer.guide_width
            center_yn = (y + block_height / 2) / \
                self.__coordinateTransformer.guide_height
            widthn = block_width / self.__coordinateTransformer.guide_width
            heightn = block_height / self.__coordinateTransformer.guide_height

            data = YoloLabel(
                clazz=to_ML_nbr(block.__class__),
                x_center=center_xn,
                y_center=center_yn,
                w=widthn,
                h=heightn
            )
            block_labels.append(data)

            # drawer = ImageDraw(grid_image)
            # left,right = np.array([ center_xn - widthn/2,center_xn + widthn/2])*self.__coordinateTransformer.guide_width
            # top,bottom  = np.array([center_yn - heightn/2, center_yn + heightn/2])*self.__coordinateTransformer.guide_height
            # drawer.rectangle(((left,top),(right,bottom)),width=2)

        return YoloSample(block_labels, image=grid_image.resize(new_size))

    def generate(self):
        filenames = listdir(self.__source_samples_folder)
        validation_and_train, test = train_test_split(filenames, test_size=0.1)
        train, validation = train_test_split(
            validation_and_train, test_size=0.2)

        for sample_set, folder_name in [
                (train, "train"), (validation, "val"), (test, "test")]:

            for filename in tqdm(sample_set, desc=f"Creating {folder_name}"):
                file_path: Path = self.__source_samples_folder / filename
                stem = file_path.stem
                sample = None

                with open(file_path, "rb") as f:
                    grid = pickle.load(f)
                    sample = self.__generate_sample(grid)

                label_path = self.__destination_labels_folder / \
                    folder_name / (stem + ".txt")

                with open(label_path, "w", encoding="utf8") as f:
                    f.write(str(sample))

                image_path = self.__destination_image_folder / \
                    folder_name / (stem + ".png")
                sample.image.save(image_path)


class AddGridLabel():
    def __init__(self, old_labels_path: Path, new_labels_path: Path,
                 x1: int, x2: int, y1: int, y2: int, img_width: int, img_height: int) -> None:

        img = Image.open(
            "/home/main/Desktop/asmail_kod/drag-and-drop/datasets/new_yolo_dataset/images/test/13.png")
        drawer = ImageDraw(img)
        drawer.rectangle((x1, y1, x2, y2), width=4)
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width / 2
        center_y = y1 + height / 2

        width_n = width / img_width
        height_n = height / img_height

        center_xn = center_x / img_width
        center_yn = center_y / img_height

        for section in ["train", "val", "test"]:
            for filename in listdir(old_labels_path / section):
                with open(old_labels_path / section / filename, "r") as rf, open(new_labels_path / section / filename, "w") as wf:
                    grid_label = "{} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(
                        ML_STR_TO_ML_NBR["GRID"],
                        center_xn,
                        center_yn,
                        width_n,
                        height_n
                    )

                    wf.writelines([grid_label] + rf.readlines())
