from typing import TypeGuard
from collections.abc import Collection
from tools.repsentantions.asObjects.definitions.grid import Grid
from tools.repsentantions.asObjects.utils.block.typeguards import is_Block


def is_grid(coll: Collection) -> TypeGuard[Grid]:
    if isinstance(coll, frozenset) and all(
            [is_Block(block) for block in coll]):
        return True
    return False
