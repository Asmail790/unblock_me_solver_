from tools.grid.utils.convience import create_grid_with_dict,create_grid, move_block_by_delta_x

from tools.position.definition import Position as Pos
from tools.block.definition import VerticalMovableBlock2X as V2X
from tools.block.definition import VerticalMovableBlock3X as V3X
from tools.block.definition import HorizontalMovableBlock3X as H3X
from tools.block.definition import HorizontalMovableBlock2X as H2X
from tools.block.definition import FixedBlock as Fixed
from tools.block.definition import Block
from tests.calc1 import calculate_occupied_positions as calc_occupied, update_occupied_by_block_for_new_block_only as update_occupied
from tests.calc2 import calculate_positions_reachable_by_block as calc_reach
from tests.calc2 import update_positions_reachable_by_block_for_new_block_only as update_reach
from tests.calc2 import update_positions_reachable_by_block_for_effected_blocks as update_reach_effected
from tools.grid.utils.convience import ALL_GRID_POSITIONS
from typing import MutableMapping

from tests.calc1 import calculate_occupied_positions as calc_occupied
from tests.calc1 import update_occupied_by_block_for_new_block_only as update_occupied

from tests.calc2 import calculate_positions_reachable_by_block as calc_reach
from tests.calc2 import update_positions_reachable_by_block_for_new_block_only as update_reach
from tests.calc2 import update_positions_reachable_by_block_for_effected_blocks as update_reach_effected

from tests.calc3 import calculate_first_unreachable_positions_by_block_due_being_blocked as calc_unreachable
from tests.calc3 import update_first_unreachable_positions_by_block_due_being_blocked_for_new_block_only as update_unreachable
from tests.calc3 import update_positions_reachable_by_block_for_effected_blocks as update_unreachable_effected

from tests.calc4 import calculate_positions_can_be_occupied_by_block as calc_positions_can_occupied
from tests.calc4 import calculate_positions_can_be_occupied_by_block_for_effected_blocks as update_positions_can_occupied_effcted
from tests.calc4 import calculate_positions_can_be_occupied_by_block_for_new_block_only as update_positions_can_occupied





def test1():
    old_block = H2X(Pos(1,0))
    old_grid1 = create_grid(V2X(Pos(0,1)),old_block,V2X(Pos(1,3)))
    old_occupied1 = calc_occupied(old_grid1)
    old_calculated_reach = calc_reach(old_grid1, old_occupied1)
    old_can_be_occupied_by = calc_positions_can_occupied(old_calculated_reach)
    correct_old_can_be_occupied_by = {
        Pos(0,0):{V2X(Pos(0,1)),H2X(Pos(1,0))},
        Pos(0,3):{V2X(Pos(0,1))},
        Pos(0,4):{V2X(Pos(0,1))},
        Pos(0,5):{V2X(Pos(0,1))},

        Pos(3,0):{H2X(Pos(1,0))},
        Pos(4,0):{H2X(Pos(1,0))},
        Pos(5,0):{H2X(Pos(1,0))},

        Pos(1,5):{V2X(Pos(1,3))},
        Pos(1,2):{V2X(Pos(1,3))},
        Pos(1,1):{V2X(Pos(1,3))},
    }


    for pos in correct_old_can_be_occupied_by.keys():
        assert correct_old_can_be_occupied_by[pos] == old_can_be_occupied_by[pos]

    new_block = move_block_by_delta_x(old_block,-1)
    new_block1 = move_block_by_delta_x(H2X(Pos(1,0)),-1)
    new_grid1 = create_grid(V2X(Pos(0,1)),new_block1,V2X(Pos(1,3)))
    new_occupied1 = update_occupied(old_block,new_block1,old_occupied1)
    new_reach = update_reach(old_block,new_block,new_occupied1,old_calculated_reach)



    effected_blocks:list[Block] = [V2X(Pos(0,1)),V2X(Pos(1,3))]
    new_reach = update_reach_effected(effected_blocks,new_occupied1,new_reach)
    new_can_be_occupied_by = update_positions_can_occupied(old_block,new_block,new_reach,old_calculated_reach,old_can_be_occupied_by)

    new_can_be_occupied_by = update_positions_can_occupied_effcted(effected_blocks,new_reach, old_calculated_reach, new_can_be_occupied_by )

    new_correct1 = {
        Pos(0,0):set(),
        Pos(0,3):{V2X(Pos(0,1))},
        Pos(0,4):{V2X(Pos(0,1))},
        Pos(0,5):{V2X(Pos(0,1))},

        Pos(2,0):{H2X(Pos(0,0))},
        Pos(3,0):{H2X(Pos(0,0))},
        Pos(4,0):{H2X(Pos(0,0))},
        Pos(5,0):{H2X(Pos(0,0))},

        Pos(1,5):{V2X(Pos(1,3))},
        Pos(1,2):{V2X(Pos(1,3))},
        Pos(1,1):{V2X(Pos(1,3))},
        
        }
    
    for pos in new_correct1.keys():
        assert new_correct1[pos] == new_can_be_occupied_by[pos]

  
      

    