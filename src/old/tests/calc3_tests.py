
from tools.grid.definition import Grid
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

def test1():
    block1 = H2X(Pos(0,0))
    grid1 = create_grid( H2X(Pos(0,0)))
    occupied1 = calc_occupied(grid1)
    correct1 = {H2X(Pos(0,0)):set()}
    calc_unreachable(grid1, occupied1)

    assert correct1[block1] == correct1[block1]


    block2 = V2X(Pos(0,0))
    grid2 = create_grid(block2)
    occupied2 = calc_occupied(grid2)
    correct2 = {V2X(Pos(0,0)):set()}
    calculated_uncreacheble2 = calc_unreachable(grid2, occupied2)

    assert correct2[block2] == calculated_uncreacheble2[block2]

    grid3 = create_grid(Fixed(Pos(0,0)),H2X(Pos(1,0)))
    occupied3 = calc_occupied(grid3)
    correct3 = {H2X(Pos(1,0)):{Pos(0,0)}, Fixed(Pos(0,0)):set()}
    calculated_uncreacheble3 = calc_unreachable(grid3, occupied3)

    assert correct3[H2X(Pos(1,0))] ==calculated_uncreacheble3[H2X(Pos(1,0))]
    assert correct3[Fixed(Pos(0,0))] ==calculated_uncreacheble3[Fixed(Pos(0,0))]

    grid4 = create_grid(V2X(Pos(0,2)),H2X(Pos(1,2)),V2X(Pos(3,2)))
    occupied4 = calc_occupied(grid4)
    correct4 = {H2X(Pos(1,2)):{Pos(0,2),Pos(3,2)}, V2X(Pos(0,2)):set(), V2X(Pos(3,2)):set()}
    calculated_uncreacheble4 = calc_unreachable(grid4, occupied4)

    assert correct4[H2X(Pos(1,2))] ==calculated_uncreacheble4[H2X(Pos(1,2))]
    assert correct4[V2X(Pos(0,2))] ==calculated_uncreacheble4[V2X(Pos(0,2))]
    assert correct4[V2X(Pos(3,2))] ==calculated_uncreacheble4[V2X(Pos(3,2))]


def test2():
    old_block = H2X(Pos(1,0))
    old_grid1 = create_grid(V2X(Pos(0,1)),old_block,V2X(Pos(1,3)))
    old_occupied1 = calc_occupied(old_grid1)
    old_correct1 = {H2X(Pos(1,0)):set(),V2X(Pos(0,1)):set(),V2X(Pos(1,3)):{Pos(1,0)}}
    old_calculated_uncreacheble1 = calc_unreachable(old_grid1, old_occupied1)

    assert old_correct1[H2X(Pos(1,0))] ==old_calculated_uncreacheble1[H2X(Pos(1,2))]
    assert old_correct1[V2X(Pos(0,1))] ==old_calculated_uncreacheble1[V2X(Pos(0,1))]
    assert old_correct1[V2X(Pos(1,3))] ==old_calculated_uncreacheble1[V2X(Pos(1,3))]

    new_block1 = move_block_by_delta_x(H2X(Pos(1,0)),-1)
    new_grid1 = create_grid(V2X(Pos(0,1)),new_block1,V2X(Pos(1,3)))
    new_occupied1 = update_occupied(old_block,new_block1,old_occupied1)
    new_partially_calculated_uncreacheble1 = update_unreachable(old_block,new_block1,new_occupied1,old_calculated_uncreacheble1)
    new_calculated_uncreacheble1 = update_unreachable_effected([V2X(Pos(0,1))],new_occupied1,new_partially_calculated_uncreacheble1)
    new_correct1 = {H2X(Pos(0,0)):set(),V2X(Pos(0,1)):{Pos(0,0)},V2X(Pos(1,3)):{Pos(1,0)}}

    assert new_correct1[H2X(Pos(0,0))] ==new_calculated_uncreacheble1[H2X(Pos(0,0))]
    assert new_correct1[V2X(Pos(0,1))] ==new_calculated_uncreacheble1[V2X(Pos(0,1))]
    assert new_correct1[V2X(Pos(1,3))] ==new_calculated_uncreacheble1[V2X(Pos(1,3))]

    











