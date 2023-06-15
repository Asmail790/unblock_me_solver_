
from collections.abc import Set, Collection
from dataclasses import replace, dataclass
from tools.repsentantions.asObjects.definitions.positon import Position
from tools.repsentantions.asObjects.definitions.block import Block


@dataclass(frozen=True, match_args=True, slots=True)
class MovementDescription():
    block: Block
    new_position: Position


@dataclass(frozen=True, match_args=True, slots=True)
class PlacementDescription():
    new_block_added: Block
