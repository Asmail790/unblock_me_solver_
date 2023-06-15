from typing import Literal, TypeGuard
from collections.abc import Mapping
from tools.repsentantions.asObjects.definitions.positon import Position
from enum import Enum, auto

ML_ClASS_NBR = Literal[0, 1, 2, 3, 4, 5, 6]
ML_CLASS_STR = Literal["H2X", "H3X", "V2X",
                       "V3X", "MainBlock", "FixedBlock", "GRID"]

ML_NBR_TO_STR: Mapping[ML_ClASS_NBR, ML_CLASS_STR] = {
    0: "H2X",
    1: "H3X",
    2: "V2X",
    3: "V3X",
    4: "FIXED",
    5: "MAIN",
    6: "GRID",
}
ML_STR_TO_ML_NBR: Mapping[ML_CLASS_STR, ML_ClASS_NBR] = {
    v: k for k, v in ML_NBR_TO_STR.items()
}


GRID_SIZE = 6
MAINBLOCK_Y_POSITION = 2
FINISH_POSITION = Position(4, MAINBLOCK_Y_POSITION)

ALL_GRID_POSITIONS = frozenset(
    [Position(x=x, y=y)for x in range(GRID_SIZE) for y in range(GRID_SIZE)])


def is_ML_NBR(x: int) -> TypeGuard[ML_ClASS_NBR]:
    return isinstance(x, int) and 0 <= x <= 6


class Theme(Enum):
    AUTUMN = auto()
    CHERRY_WOOD = auto()
    CHINESE_NEW_YEAR = auto()
    CHOCCY_SWEETIE = auto()
    CHRISTMAS = auto()
    EASTER = auto()
    HALLOWEEN = auto()
    MAPLE_WOOD = auto()
    OAK_WOOD = auto()
    OCEAN_BLUE = auto()
    ORIGNAL_1 = auto()
    ORIGNAL_2 = auto()
    OUR_5TH_YEAR = auto()
    PINE_WOOD = auto()
    SERENE_SIAM_SONGKRAN = auto()
    SPRING = auto()
    SUMMER = auto()
    VALENTINE = auto()
    WINTER = auto()
    WOODEN_BRIGE = auto()
