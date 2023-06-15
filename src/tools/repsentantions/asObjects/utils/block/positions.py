from collections.abc import Set
from tools.repsentantions.asObjects.definitions.block import Block
from tools.repsentantions.asObjects.utils.block.typeguards import is_fixed_block, is_HBlock, is_VBlock
from tools.repsentantions.asObjects.definitions.positon import Position
from tools.common.constants import ALL_GRID_POSITIONS


def occupied_positions_by_block(block: Block) -> Set[Position]:
    x = block.pos.x
    y = block.pos.y

    if is_HBlock(block):
        return set([Position(x=x + i, y=y) for i in range(block.size)])

    elif is_VBlock(block):
        return set([Position(x=x, y=y + i) for i in range(block.size)])

    elif is_fixed_block(block):
        return set([Position(x=x, y=y)])

    raise TypeError()


def tail_of_block(block: Block):
    if is_fixed_block(block):
        return block.pos

    if is_HBlock(block):
        return Position(x=block.pos.x + block.size - 1, y=block.pos.y)

    if is_VBlock(block):
        return Position(x=block.pos.x, y=block.pos.y + block.size - 1)

    raise TypeError()


def head_of_block(block: Block):
    return block.pos


def inside_grid(block: Block) -> bool:
    return occupied_positions_by_block(block) | ALL_GRID_POSITIONS == ALL_GRID_POSITIONS
