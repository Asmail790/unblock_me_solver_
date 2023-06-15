
from tools.grid.definition import Grid
from tools.grid.utils.convience import create_grid_with_dict,create_grid, move_block_by_delta_x

from tools.position.definition import Position as Pos
from tools.block.definition import VerticalMovableBlock2X as V2X
from tools.block.definition import VerticalMovableBlock3X as V3X
from tools.block.definition import HorizontalMovableBlock2X as H2X
from tools.block.definition import HorizontalMovableBlock2X as H3X
from tools.block.definition import Block
from tests.calc1 import calculate_occupied_positions, update_occupied_by_block_for_new_block_only
from tools.grid.utils.convience import ALL_GRID_POSITIONS


def test1():
    grid =create_grid_with_dict({ 
        H2X:[
                (1,0)
            ],
        V2X:[
                (0,0),
             ]
    })

    correct_old_occupied_positions:dict[Pos,Block|None] = {
        Pos(1,0):H2X(Pos(1,0)),
        Pos(2,0):H2X(Pos(1,0)),
        Pos(0,0):V2X(Pos(0,0)),
        Pos(0,1):V2X(Pos(0,0))
    }
    for pos in ALL_GRID_POSITIONS:
        correct_old_occupied_positions[pos] = correct_old_occupied_positions.get(pos,None)
    
    position_occupied_by = calculate_occupied_positions(grid)


    for pos in ALL_GRID_POSITIONS:
        assert correct_old_occupied_positions[pos] == position_occupied_by[pos] 

def test2():
    old_block = H2X(Pos(0,0))
    old_grid =create_grid(old_block)
    
    correct_old_occupied_positions:dict[Pos,Block|None] = {
    Pos(0,0):H2X(Pos(0,0)),
    Pos(1,0):H2X(Pos(0,0)),
    }
    for pos in ALL_GRID_POSITIONS:
        correct_old_occupied_positions[pos] = correct_old_occupied_positions.get(pos,None)
    

    old_position_occupied_by = calculate_occupied_positions(old_grid)


    for pos in ALL_GRID_POSITIONS:
        assert correct_old_occupied_positions[pos] == old_position_occupied_by[pos] 

    
    new_block = move_block_by_delta_x(old_block,+1)

    new_correct_occupied_positions:dict[Pos,Block|None] = {
        Pos(1,0):H2X(Pos(1,0)),
        Pos(2,0):H2X(Pos(1,0)),
    }
    for pos in ALL_GRID_POSITIONS:
        new_correct_occupied_positions[pos] = new_correct_occupied_positions.get(pos,None)
    
    new_position_occupied_by=update_occupied_by_block_for_new_block_only(old_block,new_block,old_position_occupied_by)

    for pos in ALL_GRID_POSITIONS:
        assert new_position_occupied_by[pos] == new_position_occupied_by[pos] 
