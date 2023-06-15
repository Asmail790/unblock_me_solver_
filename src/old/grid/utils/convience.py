from termcolor import colored
from tools.constants import GRID_SIZE
from tools.position.definition import Position
from tools.grid.definition import Grid
from tools.block.definition import MainBlock
from tools.block.definition import FixedBlock
from tools.block.definition import HorizontalMovableBlock2X as H2XBlock
from tools.block.definition import HorizontalMovableBlock3X as H3XBlock
from tools.block.definition import VerticalMovableBlock2X as V2XBlock
from tools.block.definition import VerticalMovableBlock3X as V3XBlock

from tools.block.definition import Block
from tools.block.utils import occupied_positions as occupied_positions_by_block
from tools.block.utils import is_main_block
from tools.block.utils import is_vertical_movable_block
from tools.block.utils import is_horizontal_movable_block
from tools.block.utils import is_fixed_block
from tools.constants import FINISH_POSITION
from typing import Type,Sequence

from tools.position.utils import convert_to_position

from tools.grid.definition import Grid
from tools.block.utils import occupied_positions as occupied_positions_by_block
from tools.block.definition import MainBlock
from tools.position.definition import Position
from collections import defaultdict
import attrs
from typing import TypeVar, Generic



X = TypeVar("X")







ALL_GRID_POSITIONS = frozenset({Position(x=x, y=y)for x in range(GRID_SIZE) for y in range(GRID_SIZE)})
def have_no_MainBlock(grid:Grid)->bool:
    return  len(list(filter(is_main_block, grid.blocks))) == 0

def have_one_MainBlock(grid:Grid)->bool:
    return  len(list(filter(is_main_block, grid.blocks))) == 1

def have_MainBlock_at_FINISH_POSITION(grid:Grid)->bool:
    return  MainBlock(position=FINISH_POSITION) in grid.blocks

def have_MainBlock_at_FINISH_POSITION2(grid:Grid)->bool:
    for block in grid.blocks:
        if is_main_block(block) and block.position == FINISH_POSITION:
            return True 
    return False


PUZZLE_OBJECT = Grid|Block|Position|frozenset[Position|Block]


def occupied_positions(obj:PUZZLE_OBJECT)->frozenset[Position]:
   
    #TODO use functools.singledispatch 
    
    if isinstance(obj,Block):
        return occupied_positions_by_block(obj)
    elif isinstance(obj,Grid): 
        return occupied_positions_by_puzzle(obj)
    elif isinstance(obj, Position):
        return frozenset([obj])
    elif isinstance(obj,frozenset) :
         return occupied_positions_by_frozen_set(obj)

    raise ValueError()

def occupied_positions_by_puzzle(puzzle:Grid)->frozenset[Position]:
    return occupied_positions_by_frozen_set(puzzle.blocks)  
       

def occupied_positions_by_frozen_set(items:frozenset[Block|Position])->frozenset[Position]:
    occupied_positions = set()  
    for item in items:
        if (isinstance(item,Block)):
            occupied_positions |= occupied_positions_by_block(item)
        
        if (isinstance(item,Position)):
            occupied_positions |={item}
    
    return frozenset(occupied_positions)

def inside_grid(block:Block)->bool:
   return  occupied_positions(block) | ALL_GRID_POSITIONS == ALL_GRID_POSITIONS

# TODO  inside_grid(obj:Block|Position|frozenset[Position|Block])

def inside_grid2(block:Block)->bool:
    return occupied_positions(block) in ALL_GRID_POSITIONS

def overlapped_positions(obj1:PUZZLE_OBJECT, obj2:PUZZLE_OBJECT)->frozenset[Position]:
    return occupied_positions(obj1) & occupied_positions(obj2)

def have_no_overlapp(obj1:PUZZLE_OBJECT, obj2:PUZZLE_OBJECT)->bool:
    return len(overlapped_positions(obj1,obj2)) == 0

def have_overlapp(obj1:PUZZLE_OBJECT,obj2:PUZZLE_OBJECT)->bool:
    return not have_no_overlapp(obj1,obj2)

# TODO move to to block utils     
def move_positions_by_delta(positions:frozenset[Position], delta:Position)->frozenset[Position]:
    return frozenset({ attrs.evolve(position, x = position.x + delta.x , y = position.y + delta.y )  for position in positions})

# TODO move to to block utils 
def move_positions_by_delta_x(positions:frozenset[Position],delta:int)->frozenset[Position]:
    return frozenset({ attrs.evolve(position, x = position.x + delta)  for position in positions})

# TODO move to to block utils 
def move_positions_by_delta_y(positions:frozenset[Position],delta:int)->frozenset[Position]:
    return frozenset({ attrs.evolve(position, y = position.y + delta)  for position in positions})


# TODO move to to block utils move_block_by_delta_x,move_block_by_delta_y
from attrs import evolve
X = TypeVar("X", bound=Block)
def move_block_by_delta_x(block:X,x:int)->X:
    return evolve(block,position = evolve(
        block.position,
        x =  block.position.x + x
    ))

def move_block_by_delta_y(block:X,y:int)->X:
    return evolve(block,position = evolve(
        block.position,
        y =  block.position.y + y
    ))

def move_block_by_delta_x_in_grid(grid:Grid,block:Block,x:int)->Grid:
    new_block = move_block_by_delta_x(block,x)
    blocks = set(grid.blocks)
    blocks.remove(block)
    blocks.add(new_block)
    
    return Grid(frozenset(blocks))

def move_block_by_delta_y_in_grid(grid:Grid,block:Block,y:int)->Grid:
    new_block = move_block_by_delta_x(block,y)
    blocks = set(grid.blocks)
    blocks.remove(block)
    blocks.add(new_block)
    
    return Grid(frozenset(blocks))

    
ALPHABET = [char for char in "abcdefghijklmnopqrstuvwxyz"]
CAPITAL_APHABET =  [letter.capitalize() for letter in ALPHABET]
DIGTS =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
DRAW_WORLD_CHARS = DIGTS + ALPHABET + CAPITAL_APHABET 
 



def draw_puzzle(puzzle: Grid) -> str:
    world: defaultdict[Position, str] = defaultdict(lambda: "□")


    for idx, block in enumerate(puzzle.blocks):
        idx = DRAW_WORLD_CHARS[idx]
        for position in occupied_positions(block):

            if  isinstance(block,MainBlock):
                # TODO only change background color not text 
                world[position] = colored(str(idx),color="red")
            
            if  isinstance(block,H2XBlock): 
                 world[position] = colored(str(idx),color="green")
            
            if  isinstance(block,H3XBlock): 
                 world[position] = colored(str(idx),color="yellow")

            if  isinstance(block,V2XBlock): 
                 world[position] = colored(str(idx),color="magenta")

            if  isinstance(block,V3XBlock): 
                 world[position] = colored(str(idx),color="white")

            if  isinstance(block,FixedBlock): 
                 world[position] = colored(str(idx),color="blue")

    worldstr = ""
    for y in range(6):
        for x in range(6):
            worldstr += world[Position(x, y)]
        worldstr += "\n"

    return worldstr


def create_grid(*args:Block)->Grid:
    return Grid(frozenset(args))

def delete_blocks_from_grid(grid:Grid,*args:Block)->Grid:
    return Grid(frozenset(grid.blocks - frozenset(args)))

def add_blocks_to_grid(grid:Grid,*args:Block)->Grid:
    return Grid(frozenset(grid.blocks | frozenset(args)))

PositionDescription = Position|tuple[int,int]
def create_grid_with_dict(description:dict[Type[Block],Sequence[PositionDescription]]): 
    blocks = []

   
    for block_type,coordinates in description.items():
        for coordinate in coordinates:
            
            if isinstance(coordinate,Position):
                blocks.append( block_type(coordinate))
            else:
                position = convert_to_position(coordinate)
                blocks.append( block_type(position))
    return Grid(frozenset(blocks))



        
