from collections.abc import Collection
from tools.repsentantions.asObjects.definitions.grid import Grid
from tools.repsentantions.asObjects.definitions.positon import Position
from tools.repsentantions.asObjects.definitions.block import Block
from tools.repsentantions.asObjects.definitions.block import H2XBlock
from tools.repsentantions.asObjects.definitions.block import H3XBlock
from tools.repsentantions.asObjects.definitions.block import V2XBlock
from tools.repsentantions.asObjects.definitions.block import V3XBlock
from tools.repsentantions.asObjects.definitions.block import MainBlock
from tools.repsentantions.asObjects.definitions.block import FixedBlock

from tools.repsentantions.asObjects.utils.convience import occupied_positions
from tools.common.constants import ALL_GRID_POSITIONS
from tools.repsentantions.asObjects.utils.convience import have_no_overlap
from tools.repsentantions.asObjects.utils.block.positions import inside_grid


class DefaultBlockPlacer:
    """
    A naive BlockPlacer.
    Works by testing positioning every type of block on every free positions.
    """
    BLOCK_TYPES: list[type[Block]] = [
        FixedBlock,
        V3XBlock,
        V2XBlock,
        H2XBlock,
        H3XBlock,
        MainBlock
    ]

    def __call__(self, puzzle: Grid) -> frozenset[Grid]:
        positions_occupied = occupied_positions(puzzle)
        unoccupied_positions = ALL_GRID_POSITIONS - positions_occupied
        possible_placements: list[Block] = []

        for unoccupied_position in unoccupied_positions:
            possible_placements += self._possible_blocks_at(
                unoccupied_position, positions_occupied)

        new_grids: set[Grid] = {Grid(
            frozenset({*puzzle, new_block})) for new_block in possible_placements}
        return frozenset(new_grids)

    def _possible_blocks_at(self, unoccupied_position: Position,
                            positions_occupied: Collection[Position]) -> list[Block]:
        possible_blocks: list[Block] = []

        for block_type in self.BLOCK_TYPES:
            block: Block = block_type(unoccupied_position)

            if inside_grid(block) and have_no_overlap(
                    block, positions_occupied):
                possible_blocks += [block]

        return possible_blocks
