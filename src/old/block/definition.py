
from abc import ABC,abstractmethod,abstractproperty
from tools.position.definition import Position
from tools.direction import Direction


from attrs import define, field
from attrs.validators import instance_of
from typing import TypeAlias



#TODO make it abstact class 

@define(frozen=True)
class Block(ABC):
    size: int = field(init=False)
    position: Position

    def short_str(self):
        return f"B(p={self.position.short_str()})"



@define(frozen=True)
class FixedBlock(Block):
    size: int = field(init=False, default=1)

    def short_str(self)->str:
        return f"F(p={self.position.short_str()})"  

# TODO make abstract class
@define(frozen=True)
class BlockWithDirection(Block,ABC):
    direction: Direction
   
         

# TODO make only possible to x position and y default set to 2 in constructor. static type checking or attrs validation ?

@define(frozen=True)
class MainBlock(BlockWithDirection):
    size: int = field(init=False, default=2)
    direction: Direction = field(init=False, default=Direction.H)

    def short_str(self)->str:
        return  f"M(s={self.size},d={self.direction.short_str()},p={self.position.short_str()})"

@define(frozen=True)
class VerticalMovableBlock3X(BlockWithDirection):
    direction: Direction  = field(init=False, default=Direction.V)
    size: int = field(init=False, default=3)

    def short_str(self)->str:
        return  f"V3X(s={self.size},d={self.direction.short_str()},p={self.position.short_str()})"

 

@define(frozen=True)
class VerticalMovableBlock2X(BlockWithDirection):
    direction: Direction  = field(init=False, default=Direction.V)
    size: int = field(init=False, default=2)


    def short_str(self)->str:
        return  f"V2X(s={self.size},d={self.direction.short_str()},p={self.position.short_str()})"



@define(frozen=True)
class HorizontalMovableBlock3X(BlockWithDirection):
    direction: Direction = field(init=False, default=Direction.H)
    size: int = field(init=False, default=3)

    def short_str(self)->str:
        return  f"H2X(s={self.size},d={self.direction.short_str()},p={self.position.short_str()})" 

@define(frozen=True)
class HorizontalMovableBlock2X(BlockWithDirection):
    direction: Direction = field(init=False, default=Direction.H)
    size: int = field(init=False, default=2)
    
    def short_str(self)->str:
        return  f"H3X(s={self.size},d={self.direction.short_str()},p={self.position.short_str()})"

    




HorizontalMovableBlock:TypeAlias = HorizontalMovableBlock2X | HorizontalMovableBlock3X 
VerticalMovableBlock =  VerticalMovableBlock2X | VerticalMovableBlock3X 
MovableBlock:TypeAlias = MainBlock | HorizontalMovableBlock | VerticalMovableBlock 

