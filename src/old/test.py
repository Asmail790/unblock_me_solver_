import sys
sys.setrecursionlimit(10_0000)
from tools.grid.utils.convience import draw_puzzle
from data.samples import puzzle_2,advanced_puzzle
from tools.grid.utils.implementions import DefaultPuzzleSolver,OptimizedBlockMover,OptimizedBFSGraphBuilderWithEarlyStop,DefaultBLockMover,PuzzleValidator,BFSGraphBuilder,DefaultPuzzleSolvedChecker,OptimizedBFSGraphBuilder,OptimizedPuzzleSolvedChecker
from pprint import pp
from time import time
from tools.grid.definition import Grid


validator = PuzzleValidator()
mover = OptimizedBlockMover()
builder = OptimizedBFSGraphBuilderWithEarlyStop()
terminal_node_checker = OptimizedPuzzleSolvedChecker()
solution_finder = DefaultPuzzleSolver(start_node=puzzle_2,producer=mover,validator=validator,builder=builder, terminal_node_checker=terminal_node_checker)


nodes,edges = solution_finder(puzzle_2)
new_grid = puzzle_2


for idx,edge in enumerate(edges):
    print("move the", str(edge.block.__class__.__name__) ,"at position", edge.block.position, "to", edge.new_position)
    xxc = set(new_grid.blocks)
    xxc.remove(edge.block)
    xxc.add(type(edge.block)(edge.new_position))
    new_grid = Grid(frozenset(xxc))
    print(draw_puzzle(new_grid))



