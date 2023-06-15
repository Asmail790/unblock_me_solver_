

from tools.repsentantions.asObjects.definitions.grid import Grid
from tools.repsentantions.asObjects.definitions.block import MainBlock
from tools.common.constants import FINISH_POSITION
from tools.repsentantions.asObjects.utils.block.typeguards import is_main_block


def have_no_MainBlock(grid: Grid) -> bool:
    return len(list(filter(is_main_block, grid))) == 0


def have_one_MainBlock(grid: Grid) -> bool:
    return len(list(filter(is_main_block, grid))) == 1


def have_MainBlock_at_FINISH_POSITION(grid: Grid) -> bool:
    return MainBlock(pos=FINISH_POSITION) in grid
