from dataclasses import replace
from collections.abc import Sequence, Mapping, Collection

from tools.repsentantions.asObjects.definitions.grid import Grid
from tools.repsentantions.asObjects.definitions.grid import Block
from tools.repsentantions.asObjects.definitions.positon import Position


def add_block(grid: Grid, block: Block) -> Grid:
    return grid.union([block])


def add_blocks(grid: Grid, blocks: Collection[Block]) -> Grid:
    return grid.union(blocks)


def remove_block(grid: Grid, block: Block) -> Grid:
    return grid.difference([block])


def remove_blocks(grid: Grid, blocks: Collection[Block]):
    return grid.difference(blocks)


def change_block_to_new_position(
    grid: Grid,
    block: Block,
    new_pos: Position
) -> Grid:
    old_block = block
    new_block = replace(old_block, pos=new_pos)
    return (grid.difference([old_block])).union([new_block])


GridDescription = Mapping[type[Block], Collection[Position]]


def create_grid(collection: Collection[Block]) -> Grid:
    return frozenset(collection)


def create_grid_with_dict(
    description: GridDescription
) -> Grid:
    blocks = []

    for block_type, coordinates in description.items():
        for coordinate in coordinates:
            blocks.append(block_type(coordinate))

    return frozenset(blocks)
