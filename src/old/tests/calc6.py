
from collections import ChainMap
from tools.grid.utils.convience import create_grid_with_dict,create_grid, move_block_by_delta_x

from tools.position.definition import Position as Pos
from tools.block.definition import VerticalMovableBlock2X as V2X
from tools.block.definition import VerticalMovableBlock3X as V3X
from tools.block.definition import HorizontalMovableBlock2X as H2X
from tools.block.definition import HorizontalMovableBlock3X as H3X
from tools.block.definition import FixedBlock as Fixed
from tools.block.definition import Block
from tools.grid.definition import Grid
from tools.grid.utils.convience import draw_puzzle

from tools.block.utils import occupied_positions as correct_calc_occupied_positions
from tools.block.utils import occupied_positions


from tools.grid.utils.convience import ALL_GRID_POSITIONS
from typing import Any, MutableMapping,Mapping

from tools.block.utils import is_horizontal_movable_block, is_vertical_movable_block
from tools.grid.utils.interfaces import BlockMovementAction, IBlockMover

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
from tests.calc5 import update_which_blocks_a_blocks_is_dependet_on_for_new_block_only as update_dependency 
from tests.calc5 import update_which_blocks_a_blocks_is_dependet_on_for_effected_blocks as update_dependency_effected
from tools.grid.utils.convience import draw_puzzle

from collections import defaultdict

from attrs import define



def reverse_map(dependency:Mapping[Block, set[Block]])->Mapping[Block, set[Block]]:
    reversed:Mapping[Block, set[Block]] = defaultdict(set)
    
    for from_,to_collection in dependency.items():
        for to in to_collection:
            reversed[to].add(from_)

    reversed_no_default = dict(reversed)
    return reversed_no_default
    


def calc_dependency_for_all_blocks(grid:Grid)->"Information":
    occupied_positions = calc_occupied(grid)
    reachable_positions = calc_reach(grid,occupied_positions)
    position_can_be_occupied_by = calc_positions_can_occupied(reachable_positions)
    unreachable_positions =calc_unreachable(grid, occupied_positions)
    dependency = calc_dependency(grid,occupied_positions,position_can_be_occupied_by,reachable_positions,unreachable_positions)
    possilbe_movemnts = calculate_blockplacments(reachable_positions)
    reversed_depdency = reverse_map(dependency) 


    return Information(
        occupied_positions=occupied_positions,
        reachable_positions=reachable_positions,
        position_can_be_occupied_by=position_can_be_occupied_by,
        unreachable_positions=unreachable_positions,
        dependency=dependency,
        possilbe_movemnts=possilbe_movemnts,
        reversed_depdency=reversed_depdency,
    )



effected_map = defaultdict(int)
def update_dependency_for_all_blocks(
    grid:Grid,
    old_block:Block,
    new_block:Block,
    info:"Information",
    )->tuple["Information",Grid]:
    blocks = set(grid.blocks.copy())
    blocks.remove(old_block)
    blocks.add(new_block)

    new_grid = Grid(frozenset(blocks))

    dependency = info.dependency
    reachable_positions = info.reachable_positions
    unreachable_positions  = info.unreachable_positions
    position_can_be_occupied_by = info.position_can_be_occupied_by
    occupied_positions = info.occupied_positions
    block_placement = info.possilbe_movemnts

    reversed_depdency = reverse_map(dependency) #TODO make part of tasks

    
    effected_blocks:list[Block] = list(reversed_depdency.get(old_block,set())) 

    # ----------------------------------------

    new_occupied_positions = update_occupied(old_block,new_block,occupied_positions)

    new_unreachable_positions_partially = update_unreachable(old_block,new_block,new_occupied_positions,unreachable_positions)
    new_unreachable_positions = update_unreachable_effected(effected_blocks,new_occupied_positions,new_unreachable_positions_partially)

    new_reachable_positions_partially = update_reach(old_block,new_block,new_occupied_positions,reachable_positions)
    new_reachable_positions = update_reach_effected(effected_blocks,new_occupied_positions,new_reachable_positions_partially)

    new_block_placement_partially = update_blockplacments_for_new_only(old_block,new_block,new_reachable_positions,block_placement)
    new_block_placement = update_blockplacments_for_effected(effected_blocks,new_reachable_positions, new_block_placement_partially)

    new_can_be_occupied_by_partially = update_positions_can_occupied(old_block,new_block,new_reachable_positions,reachable_positions,position_can_be_occupied_by)
    new_can_be_occupied_by = update_positions_can_occupied_effcted(effected_blocks,new_reachable_positions, reachable_positions, new_can_be_occupied_by_partially )

    new_dependency_partially = update_dependency(old_block,new_block,new_occupied_positions,new_reachable_positions,new_can_be_occupied_by,new_unreachable_positions,dependency)
    new_dependency = update_dependency_effected(effected_blocks,new_occupied_positions,new_reachable_positions,new_can_be_occupied_by,new_unreachable_positions,new_dependency_partially )
    


    return Information(
        occupied_positions =new_occupied_positions,
        reachable_positions=new_reachable_positions,
        dependency = new_dependency,
        unreachable_positions=new_unreachable_positions,
        position_can_be_occupied_by=new_can_be_occupied_by,
        reversed_depdency=reversed_depdency,
        possilbe_movemnts=new_block_placement,
    ),new_grid

def calculate_blockplacments(new_reachable_positions: Mapping[Block, list[Pos]]):
    possible_new_positions:dict[Block,list[Pos]] = defaultdict(list)
    
    for block,positions in new_reachable_positions.items():
            if is_horizontal_movable_block(block):
                for position in positions:
                    assert position.x != block.position.x
                    if position.x < block.position.x:
                        possible_new_positions[block].append(position)
                    
                    elif  block.position.x < position.x:
                         possible_new_positions[block].append(Pos( position.x - block.size +1,position.y))
            
            elif is_vertical_movable_block(block):
                for position in positions:
                    assert position.y != block.position.y
                    
                    if position.y < block.position.y:
                        possible_new_positions[block].append(position)
                
                    elif  block.position.y < position.y:
                            possible_new_positions[block].append(Pos(position.x,  position.y - block.size +1))

                    
                    
    return possible_new_positions








def update_blockplacments_for_new_only(
        old_block:Block,
        new_block:Block,
        new_reachable_positions: MutableMapping[Block, set[Pos]],
        old_blockplacments:MutableMapping[Block, set[Pos]]
    )->Mapping[Block, set[Pos]]:
    
    new_dict:dict[Block,set [Pos]] = {old_block:set(),new_block:set()}
    if is_horizontal_movable_block(new_block):
        for position in new_reachable_positions[new_block]:
            if position.x < new_block.position.x:
                new_dict[new_block].add(position)
            
            elif  new_block.position.x < position.x:
                new_dict[new_block].add(Pos( position.x - new_block.size +1,position.y))

            else:
                raise ValueError()
        
    if is_vertical_movable_block(new_block):
        for position in new_reachable_positions[new_block]:
            if position.y < new_block.position.y:
                new_dict[new_block].add(position)
        
            elif  new_block.position.y < position.y:
                    new_dict[new_block].add(Pos(position.x,  position.y - new_block.size +1))

            else:
                raise ValueError()
                    
    return ChainMap(new_dict,old_blockplacments)


def update_blockplacments_for_effected(
    effcted_blocks:list[Block],
    new_reachable_positions: Mapping[Block, set[Pos]],
    old_blockplacments:MutableMapping[Block, set[Pos]],
    ):
    new_dict:dict[Block,set[Pos]] = dict()
    for block in effcted_blocks:
        new_dict[block] = set()
        positions = new_reachable_positions[block]
        if is_horizontal_movable_block(block):
            for position in positions:
                if position.x < block.position.x:
                    new_dict[block].add(position)
                
                elif  block.position.x < position.x:
                        new_dict[block].add(Pos( position.x - block.size +1,position.y))

                else:
                    raise ValueError()
        
        if is_vertical_movable_block(block):
            for position in positions:
                if position.y < block.position.y:
                    new_dict[block].add(position)
            
                elif  block.position.y < position.y:
                        new_dict[block].add(Pos(position.x,  position.y - block.size +1))

                else:
                    raise ValueError()
                    
    return ChainMap(new_dict,old_blockplacments) 



@define(frozen=True)
class Information:
    occupied_positions:MutableMapping[Pos, Block | None]
    reachable_positions:MutableMapping[Block, set[Pos]]  
    position_can_be_occupied_by: MutableMapping[Pos, set[Block]]
    unreachable_positions:MutableMapping[Block, set[Pos]] 
    dependency: MutableMapping[Block, set[Block]]
    possilbe_movemnts : MutableMapping[Block,set[Pos]] 
    reversed_depdency: MutableMapping[Block, set[Block]]  



class SmartBlockMover(IBlockMover):

    def __init__(self) -> None:
        self.__first_time =True
        self.__grid_info: dict[Grid, Information] = dict()
        self.__predesscor_of:dict[Grid,Grid] = dict()
        self.__succesor_of:defaultdict[Grid,set[Grid]] = defaultdict(set)
        self.__grid_visited:set[Grid] = set()
     
    def __call__(self, grid: Grid) -> frozenset[Grid]:
        

        if self.__first_time == True:
            self.__first_time=False
            info = calc_dependency_for_all_blocks(grid)
            self.__grid_info[grid] = info 

            self.__grid_visited.add(grid)

        else:
            old_grid=  self.__predesscor_of[grid]
            action = self.description(old_grid,grid)
            old_grid_info = self.__grid_info[old_grid]
            old_block = action.block
            new_block=type(action.block)(action.new_position)
            info,_ = update_dependency_for_all_blocks(old_grid,old_block,new_block,old_grid_info )
            self.__grid_info[grid] = info 
            self.__grid_visited.add(grid)

            #if  self.__succesor_of[old_grid].issubset(self.__grid_visited):
            #    self.__grid_info[old_grid] = self.truncate(old_grid_info)


        for block,poses in info.possilbe_movemnts.items():
            for pos in poses:
                new_block = type(block)(pos)
                blocks = set(grid.blocks)
                blocks.remove(block)
                blocks.add(new_block)
                new_grid = Grid(frozenset(blocks))
                self.__predesscor_of[new_grid]= grid
                self.__succesor_of[grid].add(new_grid)

       
            
        return frozenset( self.__succesor_of[grid]) 
    
    def truncate(self,info:Information):
        return Information(
        dependency = dict(info.dependency),
        occupied_positions = dict( info.occupied_positions),
        position_can_be_occupied_by = dict(info.position_can_be_occupied_by),
        possilbe_movemnts = dict(info.possilbe_movemnts),
        reachable_positions = dict(info.reachable_positions),
        reversed_depdency = dict(info.reversed_depdency ),
        unreachable_positions = dict( info.unreachable_positions ),
        )



    def description(self,before:Grid,after:Grid)->BlockMovementAction:
        old_block = list((before.blocks - after.blocks)) [0]
        new_block = list(( after.blocks - before.blocks )) [0]

        return BlockMovementAction(block=old_block,new_position=new_block.position)


class SmartBlockMover2(IBlockMover):
    def __call__(self, grid: Grid) -> frozenset[Grid]:
        occupied_positions = calc_occupied(grid)
        reachable_positions = calc_reach(grid,occupied_positions)
        possilbe_movemnts = calculate_blockplacments(reachable_positions)

        #allgrids = list()

        return frozenset([
             Grid(grid.blocks.difference([old_block]).union([ new_block:=type(old_block)(pos)]))
            for old_block,poses in possilbe_movemnts.items() for pos in poses
            ])

        #for block,poses in possilbe_movemnts.items():
        #
        #    for pos in poses:
        #        new_block = type(block)(pos)
        #        blocks = set(grid.blocks)
        #        grid.blocks.difference([block]).union([new_block])
        #        blocks.remove(block)
        #        blocks.add(new_block)
        #        new_grid = Grid(frozenset(blocks))
        #        allgrids.append(new_grid)
             
        return frozenset(allgrids)



    def description(self,before:Grid,after:Grid)->BlockMovementAction:

        old_block = list((before.blocks - after.blocks)) [0]
        new_block = list(( after.blocks - before.blocks )) [0]


        return BlockMovementAction(block=old_block,new_position=new_block.position)

class SmartBlockMover3(IBlockMover):


    def update_occupied(self,old_block:Block,new_block:Block,occupied:MutableMapping[Pos, Block | None]):
        updated = dict(occupied)
        for pos in occupied_positions(old_block):
            updated[pos] = None
        
        for pos in occupied_positions(new_block):
             updated[pos] = new_block
        
        return updated 

    def calculate_occupied(self,grid:Grid)->MutableMapping[Pos, Block | None]:
        occupied:Mapping[Pos, Block | None] = dict()
        for pos in ALL_GRID_POSITIONS:
            occupied[pos] = None
        
        for block in grid.blocks:
            for pos in occupied_positions(block):
                occupied[pos] = block
        
        return occupied





    def __init__(self) -> None:
        self.__occupied_dir:MutableMapping[Grid,MutableMapping[Pos, Block | None]] = dict()
        self.__sucessor_of:MutableMapping[Grid,Grid] = dict()


    def __call__(self, grid: Grid) -> frozenset[Grid]:
        if grid in self.__sucessor_of.keys():
                old_grid =  self.__sucessor_of[grid]
                old_ccupied = self.__occupied_dir[old_grid]
                action = self.description(old_grid,grid)
                new_block = type(action.block)(action.new_position)
                old_block = action.block
                
                occupied_positions  = self.update_occupied(old_block,new_block,old_ccupied)
                self.__occupied_dir[grid] = occupied_positions
        else:
            occupied_positions = self.calculate_occupied(grid)
            self.__occupied_dir[grid] = occupied_positions

        
        reachable_positions = calc_reach(grid,occupied_positions)
        possilbe_movemnts = calculate_blockplacments(reachable_positions)




        allgrids = list()
        for block,poses in possilbe_movemnts.items():

            for pos in poses:
                new_block = type(block)(pos)
                blocks = set(grid.blocks)
                blocks.remove(block)
                blocks.add(new_block)
                new_grid = Grid(frozenset(blocks))
                self.__sucessor_of[new_grid] = grid
                allgrids.append(new_grid)
             
        return frozenset(allgrids)



    def description(self,before:Grid,after:Grid)->BlockMovementAction:
        old_block = list((before.blocks - after.blocks)) [0]
        new_block = list(( after.blocks - before.blocks )) [0]


        return BlockMovementAction(block=old_block,new_position=new_block.position)
    

