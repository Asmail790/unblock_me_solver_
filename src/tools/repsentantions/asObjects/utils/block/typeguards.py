from typing import TypeGuard

from tools.repsentantions.asObjects.definitions.block import Block
from tools.repsentantions.asObjects.definitions.block import MainBlock
from tools.repsentantions.asObjects.definitions.block import H2XBlock
from tools.repsentantions.asObjects.definitions.block import H3XBlock
from tools.repsentantions.asObjects.definitions.block import V2XBlock
from tools.repsentantions.asObjects.definitions.block import V3XBlock
from tools.repsentantions.asObjects.definitions.block import FixedBlock
from tools.repsentantions.asObjects.definitions.block import VerticalMovableBlock
from tools.repsentantions.asObjects.definitions.block import HorizontalMovableBlock
from tools.repsentantions.asObjects.definitions.block import MovableBlock


def is_Block(block) -> TypeGuard[Block]:
    match block:
        case V2XBlock(_):
            return True
        case V3XBlock(_):
            return True
        case H2XBlock(_):
            return True
        case H3XBlock(_):
            return True
        case MainBlock(_):
            return True
        case FixedBlock(_):
            return True

    return False


def is_VBlock(block: Block) -> TypeGuard[VerticalMovableBlock]:
    match block:
        case V2XBlock(_):
            return True
        case V3XBlock(_):
            return True
        case _:
            return False


def is_HBlock(block: Block) -> TypeGuard[HorizontalMovableBlock]:
    match block:
        case H2XBlock(_):

            return True
        case H3XBlock(_):

            return True
        case MainBlock(_):

            return True
        case _:

            return False


def is_main_block(block: Block) -> TypeGuard[MainBlock]:
    match block:
        case MainBlock(_):
            return True
        case _:
            return False


def is_fixed_block(block: Block) -> TypeGuard[FixedBlock]:
    match block:
        case FixedBlock(_):
            return True
        case _:
            return False


def is_movable_block(block: Block) -> TypeGuard[MovableBlock]:
    return not is_fixed_block(block)
