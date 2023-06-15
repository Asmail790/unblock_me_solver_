from tools.block.definition import FixedBlock 
from tools.block.definition import MainBlock
from tools.block.definition import VerticalMovableBlock2X as V2XBlock
from tools.block.definition import VerticalMovableBlock3X as V3XBlock
from tools.block.definition import HorizontalMovableBlock2X as H2XBlock
from tools.block.definition import HorizontalMovableBlock3X as H3XBlock

from tools.direction import Direction
from tools.grid.definition import Grid
from tools.position.definition import Position as Pos

puzzle_2 = Grid(
    frozenset([
    MainBlock(Pos(0,2)),
    H3XBlock(Pos(0,0)),
    V3XBlock(Pos(3,2)),
    V2XBlock(Pos(5,1)),
    H2XBlock(Pos(1,4)),
    H2XBlock(Pos(4,5)),
]))

simple_puzzle = Grid(frozenset([
    H3XBlock(Pos(0,0)),
    MainBlock(Pos(5,1))
]))

advanced_puzzle = Grid(frozenset([
    H2XBlock(Pos(2,0)),
    H2XBlock(Pos(2,1)),
    V2XBlock(Pos(1,0)),
    V2XBlock(Pos(4,0)),
    MainBlock(Pos(0,2)),
    V2XBlock(Pos(2,2)),
    V3XBlock(Pos(3,2)),
    V3XBlock(Pos(0,3)),
    V2XBlock(Pos(2,4)),
    V2XBlock(Pos(4,4)),
    H3XBlock(Pos(3,5)),
]))
