from itertools import combinations

from tools.common.constants import GRID_SIZE
from tools.repsentantions.asObjects.definitions.grid import Grid
from tools.repsentantions.asObjects.utils.block.typeguards import is_main_block
from tools.common.constants import MAINBLOCK_Y_POSITION
from tools.common.interafaces.checkers import IPuzzleValidator
from tools.repsentantions.asObjects.utils.convience import have_overlapp


class PuzzleValidator(IPuzzleValidator[Grid]):
    """
    A simple IPuzzleValidator.
    """

    def is_valid(self, puzzle: Grid) -> bool:
        return self._have_no_block_overlap(puzzle) and self._is_all_block_inside_grid(
            puzzle) and self._have_one_MainBlock_in_valid_position(puzzle)

    def _have_no_block_overlap(self, puzzle: Grid) -> bool:
        for b1, b2 in combinations(puzzle, 2):
            if have_overlapp(b1, b2):
                return False
        return True

    def _is_all_block_inside_grid(self, puzzle: Grid) -> bool:
        for block in puzzle:
            x, y = block.pos.x, block.pos.y
            if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
                return False
        return True

    def _have_one_MainBlock_in_valid_position(self, puzzle: Grid) -> bool:
        main_block_counter = 0

        for block in puzzle:
            if is_main_block(block):
                main_block_counter += 1

                if block.pos.y != MAINBLOCK_Y_POSITION:
                    return False

                if main_block_counter > 1:
                    return False
        return True
