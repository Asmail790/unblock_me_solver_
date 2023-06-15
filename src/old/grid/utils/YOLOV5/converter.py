from torch import Tensor, where
from ultralytics.yolo.engine.results import Results

from tools.block.definition import BrownBlock, MainBlock, FixedBlock
from tools.direction import Direction
from tools.grid.definition import Grid
from tools.block.definition import Block

from tools.constants import BLOCK_TO_CL_MAP, CL_TO_BLOCK_MAP
from tools.position.definition.position import Position


class Converter():
    """
    Converts boundingboxes obtained by Ultralytics's YOLO model to a Grid. 
    """

    @classmethod
    def to_grid(cls, result: Results)->Grid:
        blocks = []

        grid_cl = where(result.boxes.cls == BLOCK_TO_CL_MAP["grid"])[0][0]
        grid_cl = int(grid_cl)
        grid_xywh: list[float] = result.boxes.xywh[grid_cl].tolist()
        grid_xyxy: list[float] = result.boxes.xyxy[grid_cl].tolist()

        for box in result.boxes:
            cl: int = int(box.cls.item())
            if cl == BLOCK_TO_CL_MAP["grid"]:
                continue

            block_xyxy: list[float] = box.xyxy.tolist()[0]

            block = cls.to_block(block_xyxy, grid_xywh, grid_xyxy, cl)

            blocks.append(block)

        return Grid(blocks=frozenset(blocks))

    @classmethod
    def to_block(cls, block_xyxy: list[float], grid_xywh: list[float], grid_xyxy: list[float], cl: int) -> Block:

        x_left_block = block_xyxy[0]
        y_top_block = block_xyxy[1]

        x_left_grid = grid_xyxy[0]
        y_top_grid = grid_xyxy[1]
        width_grid = grid_xywh[2]
        height_grid = grid_xywh[3]
        block_type = CL_TO_BLOCK_MAP[cl]

        position = cls._transform_to_logical_grid_pos(
            x_left_block,
            y_top_block,
            x_left_grid,
            y_top_grid,
            width_grid,
            height_grid
        )

        if block_type == "silver":
            return FixedBlock(position=position)

        elif block_type == "red":
            return MainBlock(position=position)

        elif block_type == "H2x":
            return BrownBlock(direction=Direction.H, size=2, position=position)

        elif block_type == "H3x":
            return BrownBlock(direction=Direction.H, size=3, position=position)

        elif block_type == "V2x":
            return BrownBlock(direction=Direction.V, size=2, position=position)

        elif block_type == "V3x":
            return BrownBlock(direction=Direction.V, size=3, position=position)

        raise ValueError("cl " + str(cl) + " is not a block")

    @classmethod
    def _transform_to_logical_grid_pos(cls, x_left_block: float, y_top_block: float, x_left_grid: float, y_top_grid: float, width_grid: float, height_grid: float) -> Position:

        block_grid_xpos = round(((x_left_block - x_left_grid)/width_grid)*6)
        block_grid_ypos = round(((y_top_block-y_top_grid)/height_grid)*6)
        logical_grid_pos = Position(x=block_grid_xpos, y=block_grid_ypos)

        return logical_grid_pos
