from collections.abc import Sequence, Set

from tools.repsentantions.asObjects.definitions.grid import Grid
from tools.repsentantions.asObjects.utils.block.positions import occupied_positions_by_block
from tools.repsentantions.asObjects.definitions.positon import Position


def occupied_positions_by_grid(grid: Grid) -> Set[Position]:
    return set(
        [pos for block in grid for pos in occupied_positions_by_block(block)])
