from collections.abc import Mapping
from tools.common.constants import ML_CLASS_STR
from tools.repsentantions.asObjects.definitions.block import Block
from tools.repsentantions.asObjects.definitions.block import FixedBlock
from tools.repsentantions.asObjects.definitions.block import MainBlock
from tools.repsentantions.asObjects.definitions.block import H2XBlock
from tools.repsentantions.asObjects.definitions.block import H3XBlock
from tools.repsentantions.asObjects.definitions.block import V2XBlock
from tools.repsentantions.asObjects.definitions.block import V3XBlock
from tools.repsentantions.asObjects.definitions.grid import Grid

ML_STR_TO_PY_BLOCK_OBJECT: Mapping[ML_CLASS_STR, type[Block]] = {
    "FixedBlock": FixedBlock,
    "MainBlock": MainBlock,
    "H2X": H2XBlock,
    "H3X": H3XBlock,
    "V2X": V2XBlock,
    "V3X": V3XBlock
}

PY_BLOCK_OBJECT_TO_ML_STR: Mapping[type[Block], ML_CLASS_STR] = {
    v: k for k, v in ML_STR_TO_PY_BLOCK_OBJECT.items()
}
