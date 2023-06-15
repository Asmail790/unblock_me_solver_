

from tools.repsentantions.asObjects.definitions.direction import Direction
from tools.repsentantions.asObjects.definitions.positon import Position
from collections.abc import Collection, Sequence


def move_by_delta(pos: Position, delta_x: int = 0, delta_y: int = 0):
    return Position(pos.x + delta_x, pos.y + delta_y)


def move_positions_by_delta(
        positions: Collection[Position], delta_x: int = 0, delta_y: int = 0) -> Sequence[Position]:
    return tuple([move_by_delta(x, delta_x, delta_y) for x in positions])
