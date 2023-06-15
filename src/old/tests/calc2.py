from collections import defaultdict,ChainMap

from tools.block.definition import Block
from tools.grid.definition import Grid
from tools.position.definition import Position as Pos


from tools.block.utils import  tail_of_block,head_of_block,head_of_block,is_movable_block
from tools.position.utils import from_position_to_end_of_grid
from tools.position.utils import from_position_to_start_of_grid
from typing import MutableMapping,Mapping,Sequence

from itertools import takewhile,chain
def calculate_positions_reachable_by_block(grid:Grid,position_occupied_by_block:MutableMapping[Pos,bool])->MutableMapping[Block,list[Pos]]:
    

    x:MutableMapping[Block,list[Pos]] = dict([ 
        (block,
            list(chain(
            takewhile(lambda x:  position_occupied_by_block[x] is False,from_position_to_end_of_grid(tail_of_block(block),direction=block.direction,include_it_self=False)),                                                               
            takewhile(lambda x:  position_occupied_by_block[x] is False,from_position_to_start_of_grid(head_of_block(block),direction=block.direction,include_it_self=False))
            ))
        )
        for block in grid.blocks if is_movable_block(block)])
    
    return x
    
    positions_reachable_by_block:MutableMapping[Block,list[Pos]] =defaultdict(list)
    for block in grid.blocks:
        if is_movable_block(block):

            pos_after_tail_of_block = from_position_to_end_of_grid(tail_of_block(block),direction=block.direction,include_it_self=False)
            pos_before_head_of_block = from_position_to_start_of_grid(head_of_block(block),direction=block.direction,include_it_self=False)
            
            for pos in pos_after_tail_of_block:
                if position_occupied_by_block[pos] is  None:
                    positions_reachable_by_block[block].append(pos)
                else:
                    break
            
            for pos in pos_before_head_of_block:
                if position_occupied_by_block[pos] is  None:
                    positions_reachable_by_block[block].append(pos)
                else:
                    break

    return positions_reachable_by_block

def update_positions_reachable_by_block_for_new_block_only(
    block_before:Block,
    block_after:Block,
    new_occupied_by_block:MutableMapping[Pos,Block|None],
    old_positions_reachable_by_block:MutableMapping[Block,set[Pos]]
    )-> MutableMapping[Block,set[Pos]]: 

    new_dict = {block_after:set(),block_before:set()}

    if is_movable_block(block_after):
        pos_after_tail_of_block = from_position_to_end_of_grid(tail_of_block(block_after),direction=block_after.direction,include_it_self=False)
        pos_before_head_of_block = from_position_to_start_of_grid(head_of_block(block_after),direction=block_after.direction,include_it_self=False)
        
        for pos in pos_after_tail_of_block:
            if new_occupied_by_block[pos] is  None:
                new_dict[block_after].add(pos)
            else:
                break
        
        for pos in pos_before_head_of_block:
            if new_occupied_by_block[pos] is  None:
                new_dict[block_after].add(pos)
            else:
                break

    if isinstance(old_positions_reachable_by_block, ChainMap):
        return  ChainMap(new_dict,*old_positions_reachable_by_block.maps)

    return ChainMap(new_dict,old_positions_reachable_by_block)

def  update_positions_reachable_by_block_for_effected_blocks(
    effected_blocks:list[Block],
    new_occupied_by_block:MutableMapping[Pos,Block|None],
    old_positions_reachable_by_block:MutableMapping[Block,set[Pos]]
)->MutableMapping[Block,set[Pos]]:
    new_dict:dict[Block,set[Pos]] = {effected_block:set() for effected_block in effected_blocks}

    for effected_block in effected_blocks:
        if is_movable_block(effected_block):
            pos_after_tail_of_block = from_position_to_end_of_grid(tail_of_block(effected_block),direction=effected_block.direction,include_it_self=False)
            pos_before_head_of_block = from_position_to_start_of_grid(head_of_block(effected_block),direction=effected_block.direction,include_it_self=False)
            
            for pos in pos_after_tail_of_block:
                if new_occupied_by_block[pos] is  None:
                    new_dict[effected_block].add(pos)
                else:
                    break
            
            for pos in pos_before_head_of_block:
                if new_occupied_by_block[pos] is  None:
                    new_dict[effected_block].add(pos)
                else:
                    break
    if isinstance(old_positions_reachable_by_block, ChainMap):
        return  ChainMap(new_dict,*old_positions_reachable_by_block.maps)
    
    return ChainMap(new_dict,old_positions_reachable_by_block)


def calculate_positions_reachable_by_block_v2(
    grid:Grid,
    occupied_positions:Mapping[Pos,Block|None]
    )->Mapping[Block,set[Pos]]:

    reachable_positions:Mapping[Block,set[Pos]] =dict()
    for block in grid.blocks:
        reachable_positions[block]=set()
        if is_movable_block(block):

            pos_after_tail_of_block = from_position_to_end_of_grid(tail_of_block(block),direction=block.direction,include_it_self=False)
            pos_before_head_of_block = from_position_to_start_of_grid(head_of_block(block),direction=block.direction,include_it_self=False)
            
            for pos in pos_after_tail_of_block:
                if occupied_positions[pos] is  None:
                    reachable_positions[block].add(pos)
                else:
                    break
            
            for pos in pos_before_head_of_block:
                if occupied_positions[pos] is  None:
                    reachable_positions[block].add(pos)
                else:
                    break

    return reachable_positions


def update_positions_reachable_by_block_for_new_block_only_v2(
    block_before:Block,
    block_after:Block,
    new_occupied_by_block:Mapping[Pos,Block|None],
    old_reachable_positions:Mapping[Block,set[Pos]]
    )-> Mapping[Block,set[Pos]]: 

    reachable_positions = dict(old_reachable_positions)
    
    del reachable_positions[block_before]
    reachable_positions[block_after] = set()

    if is_movable_block(block_after):
        pos_after_tail_of_block = from_position_to_end_of_grid(tail_of_block(block_after),direction=block_after.direction,include_it_self=False)
        pos_before_head_of_block = from_position_to_start_of_grid(head_of_block(block_after),direction=block_after.direction,include_it_self=False)
        
        for pos in pos_after_tail_of_block:
            if new_occupied_by_block[pos] is  None:
                reachable_positions[block_after].add(pos)
            else:
                break
        
        for pos in pos_before_head_of_block:
            if new_occupied_by_block[pos] is  None:
                reachable_positions[block_after].add(pos)
            else:
                break

    return reachable_positions
