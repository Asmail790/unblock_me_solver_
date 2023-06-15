from enum import Enum, auto
from PIL import Image
from PIL.Image import Image as TImage
from PIL import ImageDraw
from attrs import define
import numpy as np
import os
from tools.block.definition import BrownBlock, MainBlock, FixedBlock
from tools.direction import Direction

from tools.grid.definition import Grid
from pathlib import Path
from typing import Literal

from tools.constants import BLOCK_TO_CL_MAP


class YoloDatasetGenerator:
    """
    Creates a dataset that can be interpreted by Ultralytics.
    """
    BLOCK_RESOURCES_PATH: Path = Path(
        "resources/blocks_and_guide/")
    IMAGE_OUTPUT_PATH: Path = Path("datasets/unblockme/images/")
    LABELS_OUTPUT_PATH: Path = Path("datasets/unblockme/labels/")

    SAMPLE_COUNTER = 0

    @classmethod
    def draw_plus_sign(cls, image: TImage, center_x: int, center_y: int, scale: float = 1):
        "Draws plus sign on a image."
        # x left 20
        # x right 1040
        drawer = ImageDraw.Draw(image)
        for extra in range(10):
            for pos in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]:
                point = np.array([center_x, center_y]) + np.array(pos)*extra
                drawer.point(point.tolist(), fill=255)

    @classmethod
    def grid_data(cls, img_size: tuple[int, int]) -> str:

        width, height = img_size
        # y top 720
        # y bottom 1780
        # x left 20
        # x right 1060

        # grid_image = grid_image.crop((20,720,1060,1780)).size
        # cls.draw_plus_sign(grid_image, int(20 +(1060-20)/2), int(720 + (1780 - 720)/2) )

        center_xn = (20 + (1060-20)/2) / width
        center_yn = (720 + (1780 - 720)/2) / height
        widthn = (1060 - 20) / width
        heightn = (1780 - 720) / height
        return "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(BLOCK_TO_CL_MAP["grid"], center_xn, center_yn, widthn, heightn)

    @classmethod
    def produce_images(cls, grids: frozenset[Grid], part_of: Literal["train", "val", "test"], factor: int = 1) -> list[tuple[TImage, list[str]]]:

        resource_images = {
            "H2x2": Image.open(cls.BLOCK_RESOURCES_PATH /
                               "2x-horizontal-layer-alternative-2.png"),
            "H2x1": Image.open(cls.BLOCK_RESOURCES_PATH /
                               "2x-horizontal-layer-alternative-1.png"),
            "H3x": Image.open(cls.BLOCK_RESOURCES_PATH / "3x-horizontal-layer.png"),
            "V2x": Image.open(cls.BLOCK_RESOURCES_PATH / "2x-vertical-block.png"),
            "V3x": Image.open(cls.BLOCK_RESOURCES_PATH / "3x-vertical.png"),
            "red": Image.open(cls.BLOCK_RESOURCES_PATH / "red-block.png"),
            "background": Image.open(cls.BLOCK_RESOURCES_PATH / "background.png")

        }

        guide = Image.open(cls.BLOCK_RESOURCES_PATH / "black-pixel-guide.png")

        position_pixels = np.asarray(guide)
        position_pixels = position_pixels[:, :, 3]
        gird_positions_pixels_row, gird_to_image_position_x = np.where(
            position_pixels == 255)

        grid_to_image_position_y = sorted(
            list(set(gird_positions_pixels_row.tolist())))
        gird_to_image_position_x = sorted(
            list(set(gird_to_image_position_x.tolist())))
        grid_to_image_position = (
            gird_to_image_position_x, grid_to_image_position_y)

        return [cls._produce_image(grid, part_of, resource_images, grid_to_image_position, factor) for grid in grids]

    @classmethod
    def _produce_image(cls, grid: Grid, part_of: Literal["train", "val", "test"], resource_images: dict[str, TImage], grid_to_image_position: tuple[list[int], list[int]], factor: int = 1) -> tuple[TImage, list[str]]:

        H2x2 = resource_images["H2x2"]
        H2x1 = resource_images["H2x1"]
        H3x = resource_images["H3x"]
        V2x = resource_images["V2x"]
        V3x = resource_images["V3x"]
        red = resource_images["red"]
        background = resource_images["background"]

        gird_to_image_position_x, grid_to_image_position_y = grid_to_image_position

        grid_image = background.copy()

        data: list[str] = []

        for block in grid.blocks:
            block_image = None
            object_class = None

            if isinstance(block, MainBlock):
                block_image = red
                object_class = BLOCK_TO_CL_MAP["red"]

            if isinstance(block, FixedBlock):
                # TODO create image of silverblock
                pass

            if isinstance(block, BrownBlock):
                if block.direction == Direction.V and block.size == 2:
                    block_image = V2x
                    object_class = BLOCK_TO_CL_MAP["V2x"]

                if block.direction == Direction.H and block.size == 2:
                    block_image = H2x1
                    object_class = BLOCK_TO_CL_MAP["H2x"]

                if block.direction == Direction.V and block.size == 3:
                    block_image = V3x
                    object_class = BLOCK_TO_CL_MAP["V3x"]

                if block.direction == Direction.H and block.size == 3:
                    block_image = H3x
                    object_class = BLOCK_TO_CL_MAP["H3x"]

            if block_image is None:
                continue

            position = (
                gird_to_image_position_x[block.position.x],
                grid_to_image_position_y[block.position.y]
            )

            center_xn = (
                position[0] + block_image.size[0]/2) / grid_image.size[0]
            center_yn = (
                position[1] + block_image.size[1]/2) / grid_image.size[1]
            widthn = block_image.size[0] / grid_image.size[0]
            heightn = block_image.size[1] / grid_image.size[1]

            grid_image.paste(block_image, position, block_image)

            data.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                object_class, center_xn, center_yn, widthn, heightn))

        data.append(cls.grid_data(grid_image.size))

        image_filepath = cls.IMAGE_OUTPUT_PATH / \
            part_of / f"{cls.SAMPLE_COUNTER}.png"
        label_filepath = cls.LABELS_OUTPUT_PATH / \
            part_of / f"{cls.SAMPLE_COUNTER}.txt"

        cls.SAMPLE_COUNTER += 1

        reduced_grid_image = grid_image.reduce(factor)
        reduced_grid_image.save(image_filepath, optimize=True)
        print("\n".join(data), file=open(label_filepath, "w"))

        return grid_image, data
