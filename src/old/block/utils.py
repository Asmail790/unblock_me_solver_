from typing import TypeGuard


from tools.block.definition import Block
from tools.block.definition import MovableBlock
from tools.block.definition import HorizontalMovableBlock
from tools.block.definition import VerticalMovableBlock
from tools.block.definition import MainBlock
from tools.block.definition import FixedBlock
from tools.block.definition import BlockWithDirection

from tools.direction import Direction
from tools.position.definition import Position


def is_vertical_movable_block(block: Block) -> TypeGuard[VerticalMovableBlock]:
    if isinstance(block, BlockWithDirection):
        return block.direction == Direction.V
    else:
        return False


def is_horizontal_movable_block(
        block: Block) -> TypeGuard[HorizontalMovableBlock]:
    if isinstance(block, BlockWithDirection):
        return block.direction == Direction.H
    else:
        return False


def is_main_block(block: Block) -> TypeGuard[MainBlock]:
    return isinstance(block, MainBlock)


def is_fixed_block(block: Block) -> TypeGuard[FixedBlock]:
    return isinstance(block, FixedBlock)


def is_movable_block(block: Block) -> TypeGuard[MovableBlock]:
    return not is_fixed_block(block)


def occupied_positions(block: Block) -> frozenset[Position]:
    x = block.position.x
    y = block.position.y

    if is_horizontal_movable_block(block):
        return frozenset({Position(x=x + i, y=y) for i in range(block.size)})

    elif is_vertical_movable_block(block):
        return frozenset({Position(x=x, y=y + i) for i in range(block.size)})

    elif is_fixed_block(block):
        return frozenset({Position(x=x, y=y)})

    raise ValueError()


def tail_of_block(block: Block):
    if is_fixed_block(block):
        return block.position

    if is_horizontal_movable_block(block):
        return Position(x=block.position.x +
                        block.size - 1, y=block.position.y)

    if is_vertical_movable_block(block):
        return Position(x=block.position.x,
                        y=block.position.y + block.size - 1)

    raise ValueError()


def head_of_block(block: Block):
    return block.position
