from dataclasses import dataclass, field
from uuid import uuid4, UUID
from tools.repsentantions.asObjects.definitions.direction import Direction
from tools.repsentantions.asObjects.definitions.positon import Position
from tools.repsentantions.asObjects.definitions.block_size import SIZE


@dataclass(frozen=True, match_args=True, slots=True)
class FixedBlock():
    pos: Position
    size: SIZE = field(init=False, default=1)
    direction: Direction = field(init=False, default="NONE")
    # id: UUID = field(default_factory=uuid4, hash=False, compare=False)


@dataclass(frozen=True, match_args=True, slots=True)
class MainBlock():
    pos: Position
    size: SIZE = field(init=False, default=2)
    direction: Direction = field(init=False, default="H")
    # id: UUID = field(default_factory=uuid4, hash=False, compare=False)


@dataclass(frozen=True, match_args=True, slots=True)
class V3XBlock():
    pos: Position
    direction: Direction = field(init=False, default="V")
    size: SIZE = field(init=False, default=3)
    # id: UUID = field(default_factory=uuid4, hash=False, compare=False)


@dataclass(frozen=True, match_args=True, slots=True)
class V2XBlock():
    pos: Position
    direction: Direction = field(init=False, default="V")
    size: SIZE = field(init=False, default=2)
    # id: UUID = field(default_factory=uuid4, hash=False, compare=False)


@dataclass(frozen=True, match_args=True, slots=True)
class H3XBlock():
    pos: Position
    direction: Direction = field(init=False, default="H")
    size: SIZE = field(init=False, default=3)
    # id: UUID = field(default_factory=uuid4, hash=False, compare=False)


@dataclass(frozen=True, match_args=True, slots=True)
class H2XBlock():
    pos: Position
    direction: Direction = field(init=False, default="H")
    size: SIZE = field(init=False, default=2)
    # id: UUID = field(default_factory=uuid4, hash=False, compare=False)


"""
The attributes x and y resepents the head position.

If a block have coordiante (x,y) = (0,0)
and size of 2 with and direction of H.
Then it will occupy positions of (0,0),(1,0).

If a block have coordiante (x,y) = (0,0)
and size of 3 with and direction of H.
Then it will occupy positions of (0,0),(1,0),(2,0).

If a block have coordiante (x,y) = (0,0)
and size of 2 with and direction of V.
Then it will occupy positions of (0,0),(0,1).


If a block have coordiante (x,y) = (0,0)
and and is of type Fixed then it only occupy (0,0).
"""


HorizontalMovableBlock = H2XBlock | H3XBlock | MainBlock
VerticalMovableBlock = V2XBlock | V3XBlock
MovableBlock = HorizontalMovableBlock | VerticalMovableBlock
Block = FixedBlock | MovableBlock
