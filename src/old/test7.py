


"""
class VerticalSmartBlock(Observer):
from __future__ import annotations
from attrs import define,evolve,field
from typing import Literal
from typing import NamedTuple
from typing import TypedDict
from itertools import combinations
from time import time
from random import randint,choice


    head:Position
    direction:Direction = field(init=False, default=Direction.V)
    size:int
    beforeIt:Position|None = field(init=False,default=None)
    afterIt:Position|None = field(init=False,default=None)
    smartGrid:list[list[SmartSquare]]

    def __attrs_post_init__(self):
        if self.head.y > 0: 
            self.beforeIt = Position(self.head.x, self.head.y - 1)
        
        if self.head.y + self.size -1 < 5:
            self.afterIt = Position(self.head.x, self.head.y + self.size)


    def move_forward(self,):
        pass


    def update(self, subject: SmartSquare):
        pass


    







@define(frozen=True)
class SmartSquare(Subject):

    x:int
    y:int 
    occupied:bool

    observers:list[SmartBlock] = field(init=False,factory=list)
    
    def attach(self, observer: SmartBlock) -> None:
        
        Attach an observer to the subject.
        
        self.observers.append(observer)

    
    def detach(self, observer: SmartBlock) -> None:
        self.observers.remove(observer)
        

    
    def notify(self) -> None:
        for observer in self.observers:
            observer.update(self)"""

#----------------------------------------------------
from tests.calc1_tests import test1 as calc1_test1,test2 as calc1_test2
from tests.calc2_tests import test1 as calc2_test1, test2 as calc2_test2
from tests.calc3_tests import test1 as calc3_test1, test2 as calc3_test2

from tests.calc4_tests import test1 as calc4_test1

from tests.calc5_tests import test1 as calc5_test1,test2 as calc5_test2, test3 as calc5_test3

from tests.calc6_tests import test1 as calc6_test2,test2 as calc6_test3
from tests.calc6 import test1 as calc6_test1


calc1_test1()
calc1_test2()

calc2_test1()
calc2_test2()

calc3_test1()
calc3_test2()

calc4_test1()


calc5_test1()
calc5_test2()
calc5_test3()

calc6_test1()
calc6_test2()
calc6_test3()