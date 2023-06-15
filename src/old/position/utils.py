
from tools.position.definition import Position
from tools.direction import Direction
from tools.constants import GRID_SIZE




def convert_to_tuple(position:Position)->tuple[int,int]:
    return (position.x,position.y)

def convert_to_position(position:tuple[int,int])->Position:
    return Position(position[0],position[1])



def from_start_of_grid_to_position(position:Position, direction:Direction,include_it_self:bool):
    from_ = 0 
    if direction == direction.V:
        to = position.y + (1 if include_it_self else 0)
        return [ Position(position.x,i) for i in range(from_,to)]
    
    if direction == direction.H:
        to = position.x + (1 if include_it_self else 0)
        return [ Position(i,position.y) for i in range(from_,to)]

def from_position_to_end_of_grid(position:Position, direction:Direction,include_it_self:bool):
    to = GRID_SIZE
    if direction == direction.V:
        from_ = position.y + (1 if not include_it_self else 0)
        return [ Position(position.x,i) for i in range(from_,to)]
    
    if direction == direction.H:
        from_ = position.x + (1 if not include_it_self else 0)
        return [ Position(i,position.y) for i in range(from_,to)]

def from_end_of_grid_to_position(position:Position, direction:Direction,include_it_self:bool):
    return list(reversed(from_position_to_end_of_grid(position, direction,include_it_self)))



def from_position_to_start_of_grid(position:Position, direction:Direction,include_it_self:bool):
    return list(reversed(from_start_of_grid_to_position(position, direction,include_it_self)))


    