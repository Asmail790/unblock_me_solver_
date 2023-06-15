
from tools.position.definition import Position
"""
Global variables such as 
Python class to ML class.
"""

from typing import Type
from tools.block.definition import HorizontalMovableBlock2X
from tools.block.definition import HorizontalMovableBlock3X
from tools.block.definition import VerticalMovableBlock2X
from tools.block.definition import VerticalMovableBlock3X
from tools.block.definition import MainBlock
from tools.block.definition import FixedBlock
from tools.block.definition import Block


BLOCK_TO_CL_MAP:dict[Type[Block],int] = { clazz:idx for idx, clazz in enumerate([
            HorizontalMovableBlock2X,
            HorizontalMovableBlock3X,
            VerticalMovableBlock2X,
            VerticalMovableBlock3X,
            MainBlock,
            FixedBlock
    ])
}

GRID_CL =  len(BLOCK_TO_CL_MAP)




CL_TO_BLOCK_MAP = {
    value:key for key,value in BLOCK_TO_CL_MAP.items()
}

GRID_SIZE = 6

MAINBLOCK_Y_POSITION = 2 
FINISH_POSITION = Position(4,MAINBLOCK_Y_POSITION)


