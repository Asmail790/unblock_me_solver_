from itertools import combinations
from collections.abc import Sequence
from typing import Optional

from tools.common.constants import GRID_SIZE
from tools.repsentantions.asObjects.definitions.grid import Grid
from tools.repsentantions.asObjects.utils.block.typeguards import is_main_block
from tools.common.constants import MAINBLOCK_Y_POSITION
from tools.common.interafaces.checkers import IPuzzleValidator
from tools.repsentantions.asObjects.utils.convience import have_overlapp

from tools.repsentantions.asObjects.utils.solvers.exceptions import BlockCollisionException
from tools.repsentantions.asObjects.utils.solvers.exceptions import BlockOutSideGridException
from tools.repsentantions.asObjects.utils.solvers.exceptions import MainBlockInvalidPostionException
from tools.repsentantions.asObjects.utils.solvers.exceptions import MoreThanOneMainBlockException
from tools.repsentantions.asObjects.utils.solvers.exceptions import ZeroMainBlockException


from tools.common.interafaces.exceptions.unvalid_puzzle import UnvalidPuzzleExceptionGroup


class PuzzleValidator(IPuzzleValidator[Grid]):
    """
    A simple IPuzzleValidator.
    """

    def is_valid(self, puzzle: Grid) -> bool:

        return self.as_exceptions(puzzle) is not None

    def as_exceptions(
            self, puzzle: Grid) -> Optional[UnvalidPuzzleExceptionGroup]:
        exces = [
            * self.__find_block_overlaps(puzzle),
            * self.__find_block_outside_grid(puzzle),
            * self.__find_invalid_MainBlock_configurations(puzzle),
        ]
        if len(exces) == 0:
            return None
        else:
            return UnvalidPuzzleExceptionGroup(exces)

    def __find_block_overlaps(
            self, puzzle: Grid) -> Sequence[BlockCollisionException]:
        return [BlockCollisionException(b1, b2) for b1, b2 in combinations(
            puzzle, 2) if have_overlapp(b1, b2)]

    def __find_block_outside_grid(
            self, puzzle: Grid) -> Sequence[BlockOutSideGridException]:

        return [BlockOutSideGridException(block) for block in puzzle if not (
            0 <= block.pos.x < GRID_SIZE and 0 <= block.pos.y < GRID_SIZE)]

    def __find_invalid_MainBlock_configurations(
            self, puzzle: Grid) -> Sequence[MoreThanOneMainBlockException | ZeroMainBlockException | MainBlockInvalidPostionException]:
        main_blocks = list(filter(is_main_block, puzzle))
        if len(main_blocks) == 0:
            return [ZeroMainBlockException()]
        if 1 < len(main_blocks):
            return [MoreThanOneMainBlockException()]

        if main_blocks[0].pos.y != MAINBLOCK_Y_POSITION:
            return [MainBlockInvalidPostionException(main_blocks[0])]

        return []
