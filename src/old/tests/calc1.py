
from collections import defaultdict, ChainMap

from typing import MutableMapping,Mapping

from tools.block.definition import Block
from tools.position.definition import Position as Pos
from tools.grid.definition import Grid

from tools.block.utils import occupied_positions
from tools.grid.utils.convience import ALL_GRID_POSITIONS


def calculate_occupied_positions(grid:Grid)->MutableMapping[Pos,bool]:
    """
    Calculates occupied positions 
    """
    occupied_by_block:MutableMapping[Pos,bool]= defaultdict(lambda:False)

    return defaultdict(lambda:False,{pos:True for block in grid.blocks for pos in  occupied_positions(block)})
    
    for block in grid.blocks:
        for pos in occupied_positions(block):
            occupied_by_block[pos] = True
    
    return occupied_by_block


def update_occupied_by_block_for_new_block_only( 
        block_before:Block,
        block_after:Block,
        old_occupied_by_block:MutableMapping[Pos,Block|None] )->Mapping[Pos,Block|None]:
    
    new_change = dict()
    for position in occupied_positions(block_before):
        new_change[position] = None
    
    for position in occupied_positions(block_after):
         new_change[position] = block_after

    if isinstance(old_occupied_by_block, ChainMap):
        return  ChainMap(new_change,*old_occupied_by_block.maps)

    return ChainMap(new_change,old_occupied_by_block)