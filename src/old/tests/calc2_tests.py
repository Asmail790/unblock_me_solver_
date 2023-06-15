

from tools.grid.definition import Grid
from tools.grid.utils.convience import create_grid_with_dict,create_grid, move_block_by_delta_x

from tools.position.definition import Position as Pos
from tools.block.definition import VerticalMovableBlock2X as V2X
from tools.block.definition import VerticalMovableBlock3X as V3X
from tools.block.definition import HorizontalMovableBlock2X as H2X
from tools.block.definition import HorizontalMovableBlock3X as H3X
from tools.block.definition import FixedBlock as Fixed
from tools.block.definition import Block
from tests.calc1 import calculate_occupied_positions as calc_occupied, update_occupied_by_block_for_new_block_only as update_occupied
from tests.calc2 import calculate_positions_reachable_by_block as calc_reach
from tests.calc2 import update_positions_reachable_by_block_for_new_block_only as update_reach
from tests.calc2 import update_positions_reachable_by_block_for_effected_blocks as update_reach_effected
from tools.grid.utils.convience import ALL_GRID_POSITIONS
from typing import MutableMapping
def test1():
    grid1 = create_grid(H2X(Pos(0,0)))
    correct1:dict[Block,set[Pos]]={H2X(Pos(0,0)):{Pos(2,0), Pos(3,0), Pos(4,0), Pos(5,0)}} 
    calculated_rech1 = calc_reach(grid1,calc_occupied(grid1))
    
    grid2 = create_grid(H2X(Pos(4,0)))
    correct2:dict[Block,set[Pos]]={H2X(Pos(4,0)):{Pos(0,0), Pos(1,0), Pos(2,0), Pos(3,0)}} 
    calculated_rech2 = calc_reach(grid2,calc_occupied(grid2))

    grid3 = create_grid(V2X(Pos(0,0)))
    correct3:dict[Block,set[Pos]]={V2X(Pos(0,0)):{Pos(0,2), Pos(0,3), Pos(0,4), Pos(0,5)}} 
    calculated_rech3 = calc_reach(grid3,calc_occupied(grid3))

    grid4 = create_grid(V2X(Pos(0,4)))
    correct4:dict[Block,set[Pos]]={V2X(Pos(0,4)):{Pos(0,0), Pos(0,1), Pos(0,2), Pos(0,3)}} 
    calculated_rech4 = calc_reach(grid4,calc_occupied(grid4))

    grid5 = create_grid(V2X(Pos(2,2)))
    correct5:dict[Block,set[Pos]]={V2X(Pos(2,2)):{Pos(2,0), Pos(2,1), Pos(2,4), Pos(2,5)}} 
    calculated_rech5 = calc_reach(grid5,calc_occupied(grid5))

    grid6 = create_grid(Fixed(Pos(0,0)),H2X(Pos(1,0)),Fixed(Pos(3,0)) )
    correct6:dict[Block,set[Pos]]={Fixed(Pos(0,0)):set(),H2X(Pos(1,0)):set(), Fixed(Pos(3,0)):set()} 
    calculated_rech6 = calc_reach(grid6,calc_occupied(grid6))

    grid7 = create_grid(Fixed(Pos(0,0)),H2X(Pos(3,0)),Fixed(Pos(5,0)) )
    correct7:dict[Block,set[Pos]]={Fixed(Pos(0,0)):set(),H2X(Pos(3,0)):{Pos(1,0),Pos(2,0)}, Fixed(Pos(5,0)):set()} 
    calculated_rech7 = calc_reach(grid7,calc_occupied(grid7))

    grid8 = create_grid(V2X(Pos(0,1)),H2X(Pos(1,0)))
    correct8:dict[Block,set[Pos]]={V2X(Pos(0,1)):{Pos(0,0),Pos(0,3),Pos(0,4),Pos(0,5)},H2X(Pos(1,0)):{Pos(0,0),Pos(3,0),Pos(4,0),Pos(5,0)}} 
    calculated_rech8 = calc_reach(grid8,calc_occupied(grid8))



    tests:list[tuple[Grid,MutableMapping[Block,set[Pos]],MutableMapping[Block,set[Pos]]]] = [
        (grid1,correct1,calculated_rech1),
        (grid2,correct2,calculated_rech2),
        (grid3,correct3,calculated_rech3),
        (grid4,correct4,calculated_rech4),
        (grid5,correct5,calculated_rech5),
        (grid6,correct6,calculated_rech6),
        (grid7,correct7,calculated_rech7),
        (grid8,correct8,calculated_rech8),
    ]

    for grid,correct,calculated in tests:
        for block in grid.blocks:
            assert correct[block] == calculated[block]



def test2():
    old_block1 = H2X(Pos(0,0))
    old_grid1 = create_grid(old_block1)
    old_correct1:dict[Block,set[Pos]]={H2X(Pos(0,0)):{Pos(2,0), Pos(3,0), Pos(4,0), Pos(5,0)}} 
    old_occupied1 = calc_occupied(old_grid1)
    old_calculated_rech1 = calc_reach(old_grid1,old_occupied1)
    new_block1 =  move_block_by_delta_x(old_block1,+1)
    new_grid1 = create_grid(new_block1)
    new_occupied1 = update_occupied(old_block1,new_block1,old_occupied1)
    new_calculated_rech1 = update_reach(old_block1,new_block1,new_occupied1,old_calculated_rech1)
    new_correct1:dict[Block,set[Pos]] = {H2X(Pos(1,0)):{Pos(0,0), Pos(3,0), Pos(4,0), Pos(5,0)}}

    old_block2 = H2X(Pos(4,0))
    old_grid2 = create_grid(old_block2)
    old_occupied2 = calc_occupied(old_grid2)
    old_correct2:dict[Block,set[Pos]]={H2X(Pos(4,0)):{Pos(0,0), Pos(1,0), Pos(2,0), Pos(3,0)}} 
    old_calculated_rech2 = calc_reach(old_grid2,calc_occupied(old_grid2))
    new_block2 =  move_block_by_delta_x(old_block2,-1)
    new_grid2 = create_grid(new_block2)
    new_occupied2 = update_occupied(old_block2,new_block2,old_occupied2)
    new_calculated_rech2 = update_reach(old_block2,new_block2,new_occupied2,old_calculated_rech2)
    new_correct2:dict[Block,set[Pos]] = {H2X(Pos(3,0)):{Pos(0,0), Pos(1,0), Pos(2,0), Pos(5,0)}}

    old_block8V2X = V2X(Pos(0,1))
    old_block8H2X  = H2X(Pos(1,0))
    old_grid8 = create_grid(old_block8V2X,old_block8H2X)
    old_occupied8 = calc_occupied(old_grid8)
    old_correct8:dict[Block,set[Pos]]={V2X(Pos(0,1)):{Pos(0,0),Pos(0,3),Pos(0,4),Pos(0,5)},H2X(Pos(1,0)):{Pos(0,0),Pos(3,0),Pos(4,0),Pos(5,0)}} 
    old_calculated_rech8 = calc_reach(old_grid8,calc_occupied(old_grid8))
    new_block8H2X =  move_block_by_delta_x(old_block8H2X,-1)
    new_block8V2X =old_block8V2X
    new_grid8 = create_grid(new_block8H2X,new_block8V2X)
    new_occupied8 = update_occupied(old_block8H2X,new_block8H2X,old_occupied8)
    new_partialy_calculated_rech8 = update_reach(old_block8H2X,new_block8H2X,new_occupied8,old_calculated_rech8)
    new_calculated_rech8 = update_reach_effected([old_block8V2X],new_occupied8,new_partialy_calculated_rech8)
    new_correct8:dict[Block,set[Pos]] = {H2X(Pos(0,0)):{ Pos(2,0), Pos(3,0), Pos(4,0),Pos(5,0)},V2X(Pos(0,1)):{Pos(0,3),Pos(0,4),Pos(0,5)}}



    tests:list[tuple[Grid,MutableMapping[Block,set[Pos]],MutableMapping[Block,set[Pos]]]] = [
        (old_grid1,old_correct1,old_calculated_rech1),
        (new_grid1,new_correct1,new_calculated_rech1),

        (old_grid2,old_correct2,old_calculated_rech2),
        (new_grid2,new_correct2,new_calculated_rech2),

        (old_grid8,old_correct8,old_calculated_rech8),
        (new_grid8,new_correct8,new_calculated_rech8),
    ]

    
    for grid,correct,calculated in tests:
        for block in grid.blocks:
            assert correct[block] == calculated[block]


