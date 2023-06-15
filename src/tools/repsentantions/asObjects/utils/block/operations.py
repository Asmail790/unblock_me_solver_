from tools.repsentantions.asObjects.definitions.block import Block
from tools.repsentantions.asObjects.utils.position.operations import move_by_delta
from typing import TypeVar
from dataclasses import replace

X = TypeVar("X", bound=Block)


def safereplace(block: Block, *args, **kwargs) -> Block:
    return replace(block, *args, **{"id": block.id, **kwargs})


def move_block_by_delta(
    block: X,
    delta_x: int = 0,
    delta_y: int = 0
) -> X:
    return replace(
        block,
        position=move_by_delta(
            block.pos,
            delta_x=delta_x,
            delta_y=delta_y
        ))
