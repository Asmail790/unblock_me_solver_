

from tools.direction import Direction
from tools.block.definition import Block
from tools.block.utils import occupied_positions
from tools.position.definition import Position


def occupied_squares_assert(size: int, direction: Direction, position: Position):
    b = Block(
        direction=direction,
        position=position,
        size=size,
        color=Color.B)

    if direction == Direction.H:
        correct_occupied_squares = {
            Position(x + position.x, position.y) for x in range(size)}
    else:
        correct_occupied_squares = {
            Position(position.x, position.y + y) for y in range(size)}

    assert occupied_positions(b) == correct_occupied_squares


def test_occupied_squares():
    occupied_squares_assert(2, Direction.H, position=Position(0, 0))
    occupied_squares_assert(2, Direction.V, position=Position(2, 2))
