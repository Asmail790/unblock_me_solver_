from tools.repsentantions.asObjects.definitions.grid import Grid
from tools.repsentantions.asObjects.utils.block.typeguards import is_main_block
from tools.common.constants import FINISH_POSITION
from tools.common.interafaces.checkers import IPuzzleSolvedChecker


class DefaultPuzzleSolvedChecker(IPuzzleSolvedChecker[Grid]):
    def is_solved(self, puzzle: Grid) -> bool:

        for block in puzzle:
            if is_main_block(block):
                return block.pos == FINISH_POSITION
        return False


class OptimizedPuzzleSolvedChecker(IPuzzleSolvedChecker[Grid]):
    def is_solved(self, puzzle: Grid) -> bool:
        return any(is_main_block(block) and block.pos ==
                   FINISH_POSITION for block in puzzle)
