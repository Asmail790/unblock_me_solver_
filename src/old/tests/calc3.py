from collections import defaultdict,ChainMap

from tools.block.definition import Block
from tools.grid.definition import Grid
from tools.position.definition import Position as Pos


from tools.block.utils import  tail_of_block,head_of_block,head_of_block,is_movable_block
from tools.position.utils import from_position_to_end_of_grid
from tools.position.utils import from_position_to_start_of_grid
from typing import MutableMapping,Mapping


def calculate_first_unreachable_positions_by_block_due_being_blocked(
    grid:Grid,
    position_occupied_by_block:MutableMapping[Pos,Block|None]
    )->MutableMapping[Block,set[Pos]]:

    unreachable_positions_by_block_due_being_blocked:MutableMapping[Block,set[Pos]] =defaultdict(set)

    for block in grid.blocks:
        if is_movable_block(block):
            pos_after_tail_of_block = from_position_to_end_of_grid(tail_of_block(block),direction=block.direction,include_it_self=False)
            pos_before_head_of_block = from_position_to_start_of_grid(head_of_block(block),direction=block.direction,include_it_self=False)
            
            for pos in pos_after_tail_of_block:
                if position_occupied_by_block[pos] is not  None:
                    unreachable_positions_by_block_due_being_blocked[block].add(pos)
                    break
            
            for pos in pos_before_head_of_block:
                if position_occupied_by_block[pos] is not None:
                    unreachable_positions_by_block_due_being_blocked[block].add(pos)
                    break 

    return unreachable_positions_by_block_due_being_blocked

def update_first_unreachable_positions_by_block_due_being_blocked_for_new_block_only(
    block_before:Block,
    block_after:Block,
    new_position_occupied_by_block:MutableMapping[Pos,Block|None],
    old_unreachable_positions_by_block_due_being_blocked:MutableMapping[Block,set[Pos]]
    )->MutableMapping[Block,set[Pos]]: 

    new_dict:MutableMapping[Block,set[Pos]] = {block_before:set(),block_after:set()}

    if is_movable_block(block_after):
        pos_after_tail_of_block = from_position_to_end_of_grid(tail_of_block(block_after),direction=block_after.direction,include_it_self=False)
        pos_before_head_of_block = from_position_to_start_of_grid(head_of_block(block_after),direction=block_after.direction,include_it_self=False)
        
        for pos in pos_after_tail_of_block:
            if new_position_occupied_by_block[pos] is not  None:
                new_dict[block_after].add(pos)
                break
        
        for pos in pos_before_head_of_block:
            if new_position_occupied_by_block[pos] is not None:
                new_dict[block_after].add(pos)
                break 

    if isinstance(old_unreachable_positions_by_block_due_being_blocked, ChainMap):
        return  ChainMap(new_dict,*old_unreachable_positions_by_block_due_being_blocked.maps)
    
    return ChainMap(new_dict,old_unreachable_positions_by_block_due_being_blocked)

def  update_positions_reachable_by_block_for_effected_blocks(
    effected_blocks:list[Block],
    new_position_occupied_by_block:MutableMapping[Pos,Block|None],
    old_unreachable_positions_by_block_due_being_blocked:MutableMapping[Block,set[Pos]]
    )->MutableMapping[Block,set[Pos]]: 

    new_dict:MutableMapping[Block,set[Pos]] = { effected_block:set() for effected_block in effected_blocks}

    for effected_block in  effected_blocks:
        if is_movable_block(effected_block):
            pos_after_tail_of_block = from_position_to_end_of_grid(tail_of_block(effected_block),direction=effected_block.direction,include_it_self=False)
            pos_before_head_of_block = from_position_to_start_of_grid(head_of_block(effected_block),direction=effected_block.direction,include_it_self=False)
            
            for pos in pos_after_tail_of_block:
                if new_position_occupied_by_block[pos] is not  None:
                    new_dict[effected_block].add(pos)
                    break
            
            for pos in pos_before_head_of_block:
                if new_position_occupied_by_block[pos] is not None:
                    new_dict[effected_block].add(pos)
                    break 
   
    if isinstance(old_unreachable_positions_by_block_due_being_blocked, ChainMap):
        return  ChainMap(new_dict,*old_unreachable_positions_by_block_due_being_blocked.maps)

    return ChainMap(new_dict,old_unreachable_positions_by_block_due_being_blocked)
