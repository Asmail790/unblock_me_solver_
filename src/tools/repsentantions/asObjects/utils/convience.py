from collections.abc import Collection, Set
from typing import TypeGuard
from functools import singledispatchmethod

from tools.repsentantions.asObjects.definitions.positon import Position
from tools.repsentantions.asObjects.definitions.grid import Grid
from tools.repsentantions.asObjects.definitions.block import Block

from tools.repsentantions.asObjects.utils.block.positions import occupied_positions_by_block
from tools.repsentantions.asObjects.utils.grid.positions import occupied_positions_by_grid
from tools.repsentantions.asObjects.utils.grid.typeguards import is_grid
PUZZLE_OBJECT = Grid | Block | Collection[Position]


def occupied_positions(obj: PUZZLE_OBJECT) -> Set[Position]:

    if isinstance(obj, Block):
        return occupied_positions_by_block(obj)
    elif is_collection_of_poses(obj):
        return occupied_positions_poses(obj)
    elif is_grid(obj):
        return occupied_positions_by_grid(obj)

    raise ValueError()


def occupied_positions_poses(poses: Collection[Position]) -> Set[Position]:
    return set(poses)


def is_collection_of_poses(
        coll: Collection) -> TypeGuard[Collection[Position]]:
    return all([isinstance(item, Position) for item in coll])


def overlapped_positions(obj1: PUZZLE_OBJECT,
                         obj2: PUZZLE_OBJECT) -> Set[Position]:

    return occupied_positions(obj1) & occupied_positions(obj2)


def have_no_overlap(obj1: PUZZLE_OBJECT, obj2: PUZZLE_OBJECT) -> bool:
    return len(overlapped_positions(obj1, obj2)) == 0


def have_overlapp(obj1: PUZZLE_OBJECT, obj2: PUZZLE_OBJECT) -> bool:
    return not have_no_overlap(obj1, obj2)
