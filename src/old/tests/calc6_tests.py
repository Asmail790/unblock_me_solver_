from tests.calc6 import update_dependency_for_all_blocks as update,calc_dependency_for_all_blocks as calc


from tools.grid.utils.convience import create_grid_with_dict,create_grid, move_block_by_delta_x

from tools.position.definition import Position as Pos
from tools.block.definition import VerticalMovableBlock2X as V2X
from tools.block.definition import VerticalMovableBlock3X as V3X
from tools.block.definition import HorizontalMovableBlock3X as H3X
from tools.block.definition import HorizontalMovableBlock2X as H2X
from tools.block.definition import FixedBlock as Fixed
from tools.block.definition import MainBlock as Main
from tools.block.definition import Block
from tools.grid.definition import Grid
from tools.grid.utils.convience import draw_puzzle



def test1():

    grid = create_grid(V2X(Pos(0,0)),H2X(Pos(1,0)),H3X(Pos(3,0)))
    info = calc(grid)
    info.possilbe_movemnts
    correct_positions = {
        V2X(Pos(0,0)): {
            Pos(0,1),
            Pos(0,2),
            Pos(0,3),
            Pos(0,4)
        },
        H2X(Pos(1,0)):set(),
        H3X(Pos(3,0)):set()
    }
    assert correct_positions[ V2X(Pos(0,0))] == info.possilbe_movemnts[V2X(Pos(0,0))]

    assert correct_positions[H2X(Pos(1,0))] == info.possilbe_movemnts[H2X(Pos(1,0))]

    assert correct_positions[ H3X(Pos(3,0))] == info.possilbe_movemnts[H3X(Pos(3,0))]

    
    info,grid = update(grid,V2X(Pos(0,0)),V2X(Pos(0,4)),info)

    correct_positions = {
        V2X(Pos(0,4)): {
            Pos(0,0),
            Pos(0,1),
            Pos(0,2),
            Pos(0,3)
        },
        H2X(Pos(1,0)):{Pos(0,0)},
        H3X(Pos(3,0)):set()
    }
    assert set() == info.possilbe_movemnts[V2X(Pos(0,0))]
    assert correct_positions[V2X(Pos(0,4))] == info.possilbe_movemnts[V2X(Pos(0,4))]

    assert correct_positions[H2X(Pos(1,0))] == info.possilbe_movemnts[H2X(Pos(1,0))]

    assert correct_positions[ H3X(Pos(3,0))] == info.possilbe_movemnts[H3X(Pos(3,0))]

    info,grid = update(grid,H2X(Pos(1,0)),H2X(Pos(0,0)),info)

    correct_positions = {
        V2X(Pos(0,4)): {
            Pos(0,1),
            Pos(0,2),
            Pos(0,3)
        },
        H2X(Pos(0,0)):{Pos(1,0)},
        H3X(Pos(3,0)):{Pos(2,0)}
    }

    assert set() == info.possilbe_movemnts[V2X(Pos(0,0))]
    assert set() == info.possilbe_movemnts[H2X(Pos(1,0))]
    assert correct_positions[V2X(Pos(0,4))] == info.possilbe_movemnts[V2X(Pos(0,4))]

    assert correct_positions[H2X(Pos(0,0))] == info.possilbe_movemnts[H2X(Pos(0,0))]

    assert correct_positions[H3X(Pos(3,0))] == info.possilbe_movemnts[H3X(Pos(3,0))]


    info,grid = update(grid,H3X(Pos(3,0)),H3X(Pos(2,0)),info)

    correct_positions = {
    V2X(Pos(0,4)): {
        Pos(0,1),
        Pos(0,2),
        Pos(0,3)
    },
    H2X(Pos(0,0)):set(),
    H3X(Pos(2,0)):{Pos(3,0)}
    }

    assert set() == info.possilbe_movemnts[V2X(Pos(0,0))]
    assert set() == info.possilbe_movemnts[H2X(Pos(1,0))]
    assert set() == info.possilbe_movemnts[H3X(Pos(3,0))]

    assert correct_positions[V2X(Pos(0,4))] == info.possilbe_movemnts[V2X(Pos(0,4))]

    assert correct_positions[H2X(Pos(0,0))] == info.possilbe_movemnts[H2X(Pos(0,0))]

    assert correct_positions[H3X(Pos(2,0))] == info.possilbe_movemnts[H3X(Pos(2,0))]

def test2():
    grid = create_grid(V2X(Pos(0,0)),V2X(Pos(0,4)),H3X(Pos(1,0)),H2X(Pos(0,3)), V3X(Pos(5,1)))
    #print(draw_puzzle(grid))

    correct_positions = {
        V2X(Pos(0,0)):{Pos(0,1)},
        V2X(Pos(0,4)):set(),
        V3X(Pos(5,1)):{Pos(5,0),Pos(5,2),Pos(5,3)},
        H3X(Pos(1,0)):{Pos(2,0),Pos(3,0)},
        H2X(Pos(0,3)):{Pos(1,3),Pos(2,3),Pos(3,3)}
    }

    info = calc(grid)
    for key in correct_positions:
        assert correct_positions[key] == info.possilbe_movemnts[key]

    info,grid = update(grid, V3X(Pos(5,1)), V3X(Pos(5,0)),info)
    #print(draw_puzzle(grid))

    correct_positions = {
    V2X(Pos(0,0)):{Pos(0,1)},
    V2X(Pos(0,4)):set(),
    V3X(Pos(5,0)):{Pos(5,1),Pos(5,2),Pos(5,3)},
    H3X(Pos(1,0)):{Pos(2,0)},
    H2X(Pos(0,3)):{Pos(1,3),Pos(2,3),Pos(3,3),Pos(4,3)}
    }
    assert  set() == info.possilbe_movemnts[V3X(Pos(5,1))]
    for key in correct_positions:
        assert correct_positions[key] == info.possilbe_movemnts[key]

    info,grid = update(grid,H2X(Pos(0,3)),H2X(Pos(4,3)),info)
    #print(draw_puzzle(grid))


    correct_positions = {
    V2X(Pos(0,0)):{Pos(0,1),Pos(0,2)},
    V2X(Pos(0,4)):{Pos(0,3),Pos(0,2)},
    V3X(Pos(5,0)):set(),
    H3X(Pos(1,0)):{Pos(2,0)},
    H2X(Pos(4,3)):{Pos(0,3),Pos(1,3),Pos(2,3),Pos(3,3)}
    }
    
    assert  set() == info.possilbe_movemnts[V3X(Pos(5,1))]
    assert  set() == info.possilbe_movemnts[H2X(Pos(0,3))]

    for key in correct_positions:
        assert correct_positions[key] == info.possilbe_movemnts[key]

    info,grid = update(grid,V2X(Pos(0,0)),V2X(Pos(0,2)),info)
    #print(draw_puzzle(grid))

    correct_positions = {
    V2X(Pos(0,2)):{Pos(0,0),Pos(0,1)},
    V2X(Pos(0,4)):set(),
    V3X(Pos(5,0)):set(),
    H3X(Pos(1,0)):{Pos(0,0),Pos(2,0)},
    H2X(Pos(4,3)):{Pos(1,3),Pos(2,3),Pos(3,3)}
    }

    for key in correct_positions:
        assert correct_positions[key] == info.possilbe_movemnts[key]
    
    assert  set() == info.possilbe_movemnts[V3X(Pos(5,1))]
    assert  set() == info.possilbe_movemnts[H2X(Pos(0,3))]
    assert  set() == info.possilbe_movemnts[V2X(Pos(0,0))]

    grid = create_grid(Main(Pos(0,0))) 
    info = calc(grid)
    info, new_grid = update(grid,Main(Pos(0,0)),Main(Pos(1,0)),info)
    print(info.possilbe_movemnts[Main(Pos(1,0))])
    
    

