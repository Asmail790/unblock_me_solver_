
from tools.repsentantions.asObjects.definitions.block import Block, MainBlock
from tools.repsentantions.asObjects.utils.convience import overlapped_positions
from tools.common.constants import GRID_SIZE, MAINBLOCK_Y_POSITION
from tools.common.interafaces.exceptions.unvalid_puzzle import UnvalidPuzzleCauseException


class BlockCollisionException(UnvalidPuzzleCauseException):
    def __init__(self, b1: Block, b2: Block) -> None:
        super().__init__(
            f"{b1} and {b2} overlapp in {overlapped_positions(b1,b2)}.")


class BlockOutSideGridException(UnvalidPuzzleCauseException):
    def __init__(self, b: Block) -> None:
        super().__init__(
            f"{b} is outside the grid which have size({GRID_SIZE},{GRID_SIZE}).")


class MoreThanOneMainBlockException(UnvalidPuzzleCauseException):
    def __init__(self) -> None:
        super().__init__("There is more than one Mainblock.")


class ZeroMainBlockException(UnvalidPuzzleCauseException):
    def __init__(self) -> None:
        super().__init__("There is no Mainblock.")


class MainBlockInvalidPostionException(UnvalidPuzzleCauseException):
    def __init__(self, mainBlock: MainBlock) -> None:
        super().__init__(
            f"MainBlock y position is {mainBlock.pos.y} and not {MAINBLOCK_Y_POSITION}.")
