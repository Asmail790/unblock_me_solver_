from tools.grid.utils.convience import create_grid_with_dict,create_grid, move_block_by_delta_x

from tools.position.definition import Position as Pos
from tools.block.definition import VerticalMovableBlock2X as V2X
from tools.block.definition import VerticalMovableBlock3X as V3X
from tools.block.definition import HorizontalMovableBlock3X as H3X
from tools.block.definition import HorizontalMovableBlock2X as H2X
from tools.block.definition import FixedBlock as Fixed
from tools.block.definition import Block
from tools.grid.definition import Grid
from tools.grid.utils.convience import draw_puzzle

from tools.block.utils import occupied_positions as correct_calc_occupied_positions

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

from tests.calc5 import calculate_which_blocks_a_block_is_dependet_on as calc_dependency
from tests.calc5 import update_which_blocks_a_blocks_is_dependet_on_for_new_block_only_v2 as update_dependency 
from tests.calc5 import update_which_blocks_a_blocks_is_dependet_on_for_effected_blocks as update_dependency_effected
from tools.grid.utils.convience import draw_puzzle


def calc_dependency_for_all_blocks(grid:Grid):
    occupied_positions = calc_occupied(grid)
    reachable_positions = calc_reach(grid,occupied_positions)
    position_can_be_occupied_by = calc_positions_can_occupied(reachable_positions)
    unreachable_positions =calc_unreachable(grid, occupied_positions)
    dependency = calc_dependency(grid,occupied_positions,position_can_be_occupied_by,reachable_positions,unreachable_positions)
    return dependency,reachable_positions,unreachable_positions,position_can_be_occupied_by,occupied_positions

def test1():
    grid1 = create_grid(V2X(Pos(0,0)))
    dp1 =  calc_dependency_for_all_blocks(grid1)[0]
   
    assert dp1[V2X(Pos(0,0))] == set()

    grid2 = create_grid(V2X(Pos(0,0)),H2X(Pos(1,0)))
    dp2 =  calc_dependency_for_all_blocks(grid2)[0]

    assert dp2[V2X(Pos(0,0))] == set()
    assert dp2[H2X(Pos(1,0))] == {V2X(Pos(0,0))}

    grid2 = create_grid(V2X(Pos(0,3)),H2X(Pos(1,0)))
    dp2 =  calc_dependency_for_all_blocks(grid2)[0]

    assert dp2[V2X(Pos(0,3))] == {H2X(Pos(1,0))}
    assert dp2[H2X(Pos(1,0))] == {V2X(Pos(0,3))}

    grid3 = create_grid(V2X(Pos(0,0)),H2X(Pos(1,0)),H2X(Pos(3,0)))
    dp3 =  calc_dependency_for_all_blocks(grid3)[0]

    assert dp3[H2X(Pos(3,0))] == {H2X(Pos(1,0))}
    assert dp3[H2X(Pos(1,0))]  == {V2X(Pos(0,0)),H2X(Pos(3,0))}
    assert dp3[V2X(Pos(0,0))] == set()


    grid4 = create_grid(V2X(Pos(0,1)),V2X(Pos(0,4)),V2X(Pos(2,4)),V2X(Pos(3,4)),V2X(Pos(4,4)),H2X(Pos(4,0)), H2X(Pos(0,3)))
    dp4 =  calc_dependency_for_all_blocks(grid4)[0]

    assert dp4[V2X(Pos(0,1))] == {H2X(Pos(4,0)),H2X(Pos(0,3))}
    assert dp4[V2X(Pos(0,4))] == {H2X(Pos(0,3))}

    assert dp4[V2X(Pos(2,4))] == { H2X(Pos(0,3)),H2X(Pos(4,0)) }   
    assert dp4[V2X(Pos(3,4))] == { H2X(Pos(0,3)),H2X(Pos(4,0)) }
    assert dp4[V2X(Pos(4,4))] == { H2X(Pos(0,3)),H2X(Pos(4,0)) }

    assert dp4[H2X(Pos(0,3))] == {V2X(Pos(2,4)),V2X(Pos(3,4)),V2X(Pos(4,4))}
    assert dp4[H2X(Pos(4,0))] == {V2X(Pos(0,1)),V2X(Pos(2,4)),V2X(Pos(3,4))}


def test2():
    grid1 = create_grid(V2X(Pos(0,1)),H2X(Pos(1,0)))
    dependency,reachable_positions,unreachable_positions,position_can_be_occupied_by,occupied_positions = calc_dependency_for_all_blocks(grid1)
    old_block = V2X(Pos(0,1))
    new_block = V2X(Pos(0,0))
    new_dependency,new_reachable_positions,new_unreachable_positions,new_position_can_be_occupied_by,new_occupied_positions,_ ,_= update_dependency_for_all_blocks(
    grid1,
    old_block,
    new_block,
    dependency,
    reachable_positions,
    unreachable_positions,
    position_can_be_occupied_by,
    occupied_positions
    )

    for pos in ALL_GRID_POSITIONS:
        if pos in correct_calc_occupied_positions(V2X(Pos(0,0))) :
            assert new_occupied_positions[pos] == V2X(Pos(0,0))
        elif pos in correct_calc_occupied_positions(H2X(Pos(1,0))):
            assert new_occupied_positions[pos] == H2X(Pos(1,0))
        else:
            assert new_occupied_positions[pos] == None

    correct_positions_that_can_be_occupied = {Pos(0,2),Pos(0,3),Pos(0,4),Pos(0,5),Pos(3,0),Pos(4,0),Pos(5,0)}
    correct_positions_that_no_can_occupy = ALL_GRID_POSITIONS - correct_positions_that_can_be_occupied
    for pos in correct_positions_that_no_can_occupy:
        assert new_position_can_be_occupied_by[pos] == set()

    for i in range(2,6):
        assert new_position_can_be_occupied_by[Pos(0,i)] == {V2X(Pos(0,0))} 

    for i in range(3,6):
        assert new_position_can_be_occupied_by[Pos(i,0)] == {H2X(Pos(1,0))} 

    assert new_unreachable_positions[V2X(Pos(0,0))] == set() 
    assert new_unreachable_positions[V2X(Pos(0,1))] == set() 
    assert new_unreachable_positions[H2X(Pos(1,0))] == {Pos(0,0)}

    assert new_reachable_positions[V2X(Pos(0,0))] == { Pos(0,i) for i in range(2,6)}
    assert new_reachable_positions[H2X(Pos(1,0))] == {Pos(i,0) for i in range(3,6)}
    assert new_reachable_positions[V2X(Pos(0,1))] == set()

    assert new_dependency[V2X(Pos(0,0))] == set()
    assert new_dependency[V2X(Pos(0,1))] == set()
    assert new_dependency[H2X(Pos(1,0))] == {V2X(Pos(0,0))}


def test3():
    grid = create_grid(
        H2X(Pos(1,0)),
        H2X(Pos(4,0)),
        V2X(Pos(0,1)),
        V2X(Pos(3,1)),
        V2X(Pos(5,1)),
        V2X(Pos(5,3)),
        )
    dependency,reachable_positions,unreachable_positions,position_can_be_occupied_by,occupied_positions = calc_dependency_for_all_blocks(grid)
    
    assert dependency[H2X(Pos(1,0))] == { V2X(Pos(0,1)),  H2X(Pos(4,0)),  V2X(Pos(3,1))}
    assert dependency[H2X(Pos(4,0))] == { H2X(Pos(1,0)), V2X(Pos(3,1))}
    assert dependency[V2X(Pos(0,1))] == { H2X(Pos(1,0))}
    assert dependency[V2X(Pos(3,1))] == { H2X(Pos(1,0)),  H2X(Pos(4,0))}
    assert dependency[V2X(Pos(5,1))] == { H2X(Pos(4,0)),V2X(Pos(5,3))}
    assert dependency[V2X(Pos(5,3))] == {V2X(Pos(5,1))}
    
    #print(draw_puzzle(grid))
    old_block = H2X(Pos(1,0))
    new_block = H2X(Pos(0,0))


    new_dependency,_2,_3,_4,_5,new_grid,_6 = update_dependency_for_all_blocks( grid,old_block,new_block,dependency,reachable_positions,unreachable_positions,position_can_be_occupied_by,occupied_positions)
    
    #print(draw_puzzle(new_grid))
    assert new_dependency[H2X(Pos(0,0))] == {H2X(Pos(4,0)),V2X(Pos(3,1))}
    assert new_dependency[H2X(Pos(4,0))] == {H2X(Pos(0,0)),V2X(Pos(3,1))}
    assert new_dependency[V2X(Pos(0,1))] == {H2X(Pos(0,0))}
    assert new_dependency[V2X(Pos(3,1))] == {H2X(Pos(0,0)),H2X(Pos(4,0))}
    assert new_dependency[V2X(Pos(5,1))] == {H2X(Pos(4,0)),V2X(Pos(5,3))}
    assert new_dependency[V2X(Pos(5,3))] == {V2X(Pos(5,1))}
    
    old_block = H2X(Pos(4,0))
    new_block = H2X(Pos(2,0))
    
    new_dependency,_2,_3,_4,_5,new_grid,_6 = update_dependency_for_all_blocks(new_grid, old_block,new_block, new_dependency,_2,_3,_4,_5)
    
    #print(draw_puzzle(new_grid))
    assert new_dependency[H2X(Pos(0,0))] == {H2X(Pos(2,0))}
    assert new_dependency[H2X(Pos(2,0))] == {H2X(Pos(0,0)),V2X(Pos(5,1))}
    assert new_dependency[V2X(Pos(0,1))] == {H2X(Pos(0,0))}
    assert new_dependency[V2X(Pos(3,1))] == {H2X(Pos(2,0))}
    assert new_dependency[V2X(Pos(5,1))] == {H2X(Pos(2,0)),V2X(Pos(5,3))}
    assert new_dependency[V2X(Pos(5,3))] == {V2X(Pos(5,1))}
    
    
    old_block = V2X(Pos(5,1))
    new_block = V2X(Pos(5,0))
    
    new_dependency,_2,_3,_4,_5,new_grid,_6 = update_dependency_for_all_blocks(new_grid, old_block,new_block, new_dependency,_2,_3,_4,_5)
    
    assert new_dependency[H2X(Pos(0,0))] == {H2X(Pos(2,0))}
    assert new_dependency[H2X(Pos(2,0))] == {H2X(Pos(0,0)),V2X(Pos(5,0))}
    assert new_dependency[V2X(Pos(0,1))] == {H2X(Pos(0,0))}
    assert new_dependency[V2X(Pos(3,1))] == {H2X(Pos(2,0))}
    assert new_dependency[V2X(Pos(5,0))] == {V2X(Pos(5,3))}
    assert new_dependency[V2X(Pos(5,3))] == {V2X(Pos(5,0))}
    
    #print(draw_puzzle(new_grid)) 



from collections import defaultdict
def reverse_map(dependency:MutableMapping[Block, set[Block]])->MutableMapping[Block, set[Block]]:
    reversed:MutableMapping[Block, set[Block]] = defaultdict(set)
    
    for from_,to_collection in dependency.items():
        for to in to_collection:
            reversed[to].add(from_)

    reversed_no_default = dict(reversed)
    return reversed_no_default
    


def update_dependency_for_all_blocks(
    grid:Grid,
    old_block:Block,
    new_block:Block,
    dependency:MutableMapping[Block, set[Block]],
    reachable_positions:MutableMapping[Block, set[Pos]],
    unreachable_positions:MutableMapping[Block, set[Pos]],
    position_can_be_occupied_by:MutableMapping[Pos, set[Block]],
    occupied_positions:MutableMapping[Pos, Block | None]
    ):
    blocks = set(grid.blocks.copy())
    blocks.remove(old_block)
    blocks.add(new_block)

    new_grid = Grid(frozenset(blocks))
    reversed_depdency = reverse_map(dependency) #TODO make part of tasks
  
    effected_blocks = list(reversed_depdency[old_block]) 


    # ----------------------------------------

    new_occupied_positions = update_occupied(old_block,new_block,occupied_positions)

    new_unreachable_positions_partially = update_unreachable(old_block,new_block,new_occupied_positions,unreachable_positions)
    new_unreachable_positions = update_unreachable_effected(effected_blocks,new_occupied_positions,new_unreachable_positions_partially)
    
    new_reachable_positions_partially = update_reach(old_block,new_block,new_occupied_positions,reachable_positions)
    new_reachable_positions = update_reach_effected(effected_blocks,new_occupied_positions,new_reachable_positions_partially)
    
    new_can_be_occupied_by_partially = update_positions_can_occupied(old_block,new_block,new_reachable_positions,reachable_positions,position_can_be_occupied_by)
    new_can_be_occupied_by = update_positions_can_occupied_effcted(effected_blocks,new_reachable_positions, reachable_positions, new_can_be_occupied_by_partially )

    new_dependency_partially = update_dependency(old_block,new_block,new_occupied_positions,new_reachable_positions,new_can_be_occupied_by,new_unreachable_positions,dependency)
    new_dependency = update_dependency_effected(effected_blocks,new_occupied_positions,new_reachable_positions,new_can_be_occupied_by,new_unreachable_positions,new_dependency_partially )

    return new_dependency,new_reachable_positions,new_unreachable_positions,new_can_be_occupied_by,new_occupied_positions, new_grid,reversed_depdency