
from collections import defaultdict,ChainMap
from typing import MutableMapping,Mapping

from tools.block.definition import Block
from tools.position.definition import Position as Pos
from tools.grid.definition import Grid


# fixa 
def calculate_which_blocks_a_block_is_dependet_on(
        grid:Grid,
        occupied_positions:MutableMapping[Pos,Block|None],
        positions_can_be_occupied_by_block:MutableMapping[Pos,set[Block]],
        positions_reachable_by_block:MutableMapping[Block,set[Pos]],
        positions_should_be_reachble_if_not_blocked:MutableMapping[Block,set[Pos]]
    )->MutableMapping[Block,set[Block]]:

    block_dependet_on:defaultdict[Block,set[Block]] = defaultdict(set)
    for block in grid.blocks:
        for position_reachable_by_block in positions_reachable_by_block[block]:
            for possible_occupier in positions_can_be_occupied_by_block[position_reachable_by_block]:
                
                # It is apparent if block moves to new position then it can not  occupy same positions as before. 
                if block == possible_occupier:
                    continue

                block_dependet_on[block].add(possible_occupier)

        
        for first_unreachable_position_due_being_blocked in positions_should_be_reachble_if_not_blocked[block]:
            block_which_occupies_position = occupied_positions[first_unreachable_position_due_being_blocked]
            if block_which_occupies_position is None or block_which_occupies_position == block:
                  raise ValueError()
            
            block_dependet_on[block].add(block_which_occupies_position)

    return block_dependet_on


def update_which_blocks_a_blocks_is_dependet_on_for_new_block_only(
        block_before:Block,
        block_after:Block,
        new_occupied_positions:MutableMapping[Pos,Block|None],
        new_positions_reachable_by_block:MutableMapping[Block,set[Pos]],
        new_positions_can_be_occupied_by_block:MutableMapping[Pos,set[Block]],
        positions_should_be_reachble_if_not_blocked:MutableMapping[Block,set[Pos]],
        old_which_blocks_a_blocks_is_dependet_on:MutableMapping[Block,set[Block]],
    )->ChainMap[Block,set[Block]]:

    new_dict:dict[Block,set[Block]] = dict()

    new_dict[block_before] = set()
    new_dict[block_after] = set()

    for position_reachable_by_block in new_positions_reachable_by_block[block_after]:
        for possible_occupier in new_positions_can_be_occupied_by_block[position_reachable_by_block]:
            
            # It is apparent if block moves to new position then it can not  occupy same positions as before. 
            if block_after == possible_occupier:
                continue

            new_dict[block_after].add(possible_occupier)
    
    
    for first_unreachable_position_due_being_blocked in positions_should_be_reachble_if_not_blocked[block_after]:
        block_which_occupies_position = new_occupied_positions[first_unreachable_position_due_being_blocked]
        
        if block_which_occupies_position is None or block_which_occupies_position == block_after:
            raise ValueError()
            
        new_dict[block_after].add(block_which_occupies_position)

    if isinstance(old_which_blocks_a_blocks_is_dependet_on, ChainMap):
        return  ChainMap(new_dict,*old_which_blocks_a_blocks_is_dependet_on.maps)

    return ChainMap(new_dict, old_which_blocks_a_blocks_is_dependet_on)


def update_which_blocks_a_blocks_is_dependet_on_for_effected_blocks(
    effected_blocks:list[Block],
    new_occupied_positions:MutableMapping[Pos,Block|None],
    new_positions_reachable_by_block:MutableMapping[Block,set[Pos]],
    new_positions_can_be_occupied_by_block:MutableMapping[Pos,set[Block]],
    positions_should_be_reachble_if_not_blocked:MutableMapping[Block,set[Pos]],
    old_which_blocks_a_blocks_is_dependet_on:MutableMapping[Block,set[Block]],
    )->MutableMapping[Block,set[Block]]:

    new_dict:dict[Block,set[Block]] = dict()

    for block in effected_blocks:
        new_dict[block] = set()

        for position_reachable_by_block in new_positions_reachable_by_block[block]:
            for possible_occupier in new_positions_can_be_occupied_by_block[position_reachable_by_block]:
                
                # It is apparent if block moves to new position then it can not  occupy same positions as before. 
                if block == possible_occupier:
                    continue

                new_dict[block].add(possible_occupier)

        
        for first_unreachable_position_due_being_blocked in positions_should_be_reachble_if_not_blocked[block]:
            block_which_occupies_position = new_occupied_positions[first_unreachable_position_due_being_blocked]
            
            if block_which_occupies_position is None or block_which_occupies_position == block:
                raise ValueError()
            
            new_dict[block].add(block_which_occupies_position)

    if isinstance(old_which_blocks_a_blocks_is_dependet_on, ChainMap):
        return  ChainMap(new_dict,*old_which_blocks_a_blocks_is_dependet_on.maps)
    
    return ChainMap(new_dict, old_which_blocks_a_blocks_is_dependet_on) 

