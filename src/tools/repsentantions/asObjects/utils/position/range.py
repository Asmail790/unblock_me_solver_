from collections.abc import Sequence

from tools.repsentantions.asObjects.definitions.positon import Position
from tools.repsentantions.asObjects.definitions.direction import Direction
from tools.common.constants import GRID_SIZE


def from_start_of_grid_to_position(
    position: Position,
    direction: Direction,
    include_it_self: bool
) -> Sequence[Position]:
    from_ = 0

    if direction == "V":
        to = position.y + (1 if include_it_self else 0)
        return tuple([Position(position.x, i) for i in range(from_, to)])

    elif direction == "H":
        to = position.x + (1 if include_it_self else 0)
        return tuple([Position(i, position.y) for i in range(from_, to)])

    raise TypeError()


def from_position_to_end_of_grid(
    position: Position,
    direction: Direction,
    include_it_self: bool
) -> Sequence[Position]:
    to = GRID_SIZE

    if direction == "V":
        from_ = position.y + (1 if not include_it_self else 0)
        return tuple([Position(position.x, i) for i in range(from_, to)])

    elif direction == "H":
        from_ = position.x + (1 if not include_it_self else 0)
        return tuple([Position(i, position.y) for i in range(from_, to)])

    raise TypeError()


def from_end_of_grid_to_position(
    position: Position,
    direction: Direction,
    include_it_self: bool
) -> Sequence[Position]:
    return tuple(reversed(
        from_position_to_end_of_grid(position, direction, include_it_self)
    ))


def from_position_to_start_of_grid(
    position: Position,
    direction: Direction,
    include_it_self: bool
) -> Sequence[Position]:
    return tuple(reversed(
        from_start_of_grid_to_position(position, direction, include_it_self)
    ))
