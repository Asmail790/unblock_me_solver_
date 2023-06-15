
from collections import defaultdict, ChainMap

from tools.block.definition import Block
from tools.position.definition import Position as Pos
from typing import MutableMapping,Mapping,Sequence
from tools.grid.utils.convience import ALL_GRID_POSITIONS

def calculate_positions_can_be_occupied_by_block(
    positions_reachable_by_block:MutableMapping[Block,set[Pos]],
    )->MutableMapping[Pos,set[Block]]:

    positions_can_be_occupied_by_block:MutableMapping[Pos,set[Block]] = defaultdict(set)

    for block,positions  in positions_reachable_by_block.items():
        for position in positions:
            positions_can_be_occupied_by_block[position].add(block)

    return positions_can_be_occupied_by_block 


def calculate_positions_can_be_occupied_by_block_for_new_block_only(
    old_block:Block,
    new_block:Block,
    new_positions_reachable_by_block:MutableMapping[Block,set[Pos]],
    old_positions_reachable_by_block:MutableMapping[Block,set[Pos]],
    old_positions_can_be_occupied_by_block:MutableMapping[Pos,set[Block]]
    )->MutableMapping[Pos,set[Block]]:
    new_dict:MutableMapping[Pos,set[Block]] = {} 

    for position in old_positions_reachable_by_block[old_block]:

        if position not in new_dict:
            new_dict[position] =  old_positions_can_be_occupied_by_block[position].copy()
        new_dict[position].remove(old_block)

    for position in new_positions_reachable_by_block[new_block]:
        
        if position not in new_dict:
            new_dict[position] =  old_positions_can_be_occupied_by_block[position].copy()
        new_dict[position].add(new_block)

    if isinstance(old_positions_can_be_occupied_by_block, ChainMap):
        return  ChainMap(new_dict,*old_positions_can_be_occupied_by_block.maps)

    return ChainMap(new_dict, old_positions_can_be_occupied_by_block)
           

def calculate_positions_can_be_occupied_by_block_for_effected_blocks(
    effected_blocks:list[Block],
    new_positions_reachable_by_block:MutableMapping[Block,set[Pos]],
    old_positions_reachable_by_block:MutableMapping[Block,set[Pos]],
    old_positions_can_be_occupied_by_block:MutableMapping[Pos,set[Block]]
)->MutableMapping[Pos,set[Block]]:
    # make new_dict a copy of old_positions_can_be_occupied_by_block for effected blocks:
    # for each effected block
    #   Positions that are both  new_positions_reachable_by_block add Block in new_dict
    #    Positions  that are only old_positions_reachable_by_block remove Block in new_dict

    new_dict:MutableMapping[Pos,set[Block]] = {}
    for effected_block in effected_blocks:
               
        for position in old_positions_reachable_by_block[effected_block]:
            if position not in new_dict: 
                new_dict[position] = old_positions_can_be_occupied_by_block[position].copy()
            new_dict[position].remove(effected_block)
        

        for position in new_positions_reachable_by_block[effected_block]:
            if position not in new_dict:
                new_dict[position] = old_positions_can_be_occupied_by_block[position].copy()
            new_dict[position].add(effected_block)

    if isinstance(old_positions_can_be_occupied_by_block, ChainMap):
        return  ChainMap(new_dict,*old_positions_can_be_occupied_by_block.maps)

    return ChainMap(new_dict,old_positions_can_be_occupied_by_block)
        
