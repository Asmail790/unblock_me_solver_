from itertools import chain, takewhile
from collections.abc import Set, Collection, Mapping, Sequence, MutableSequence, MutableMapping
from collections import defaultdict
from dataclasses import replace
from tools.common.interafaces.deriver import Deriver
from tools.repsentantions.asObjects.definitions.grid import Grid
from tools.repsentantions.asObjects.definitions.positon import Position
from tools.repsentantions.asObjects.definitions.block import Block


from tools.repsentantions.asObjects.utils.convience import occupied_positions
from tools.repsentantions.asObjects.utils.convience import have_overlapp
from tools.repsentantions.asObjects.utils.position.range import from_position_to_end_of_grid
from tools.repsentantions.asObjects.utils.position.range import from_position_to_start_of_grid

from tools.repsentantions.asObjects.utils.block.positions import head_of_block
from tools.repsentantions.asObjects.utils.block.positions import tail_of_block
from tools.repsentantions.asObjects.utils.block.operations import move_block_by_delta

from tools.repsentantions.asObjects.utils.grid.operations import change_block_to_new_position

from tools.repsentantions.asObjects.utils.block.typeguards import is_fixed_block
from tools.repsentantions.asObjects.utils.block.typeguards import is_HBlock
from tools.repsentantions.asObjects.utils.block.typeguards import is_VBlock
from tools.repsentantions.asObjects.utils.block.typeguards import is_movable_block
from tools.common.constants import GRID_SIZE


class DefaultBLockMover():

    def derive_block_movements(self, grid: Grid) -> Set[Grid]:
        result = []

        for block in grid:
            other_blocks = frozenset(
                {otherBlock for otherBlock in grid if otherBlock != block})
            occupied_positions_by_other_blocks = {
                pos for block in other_blocks for pos in occupied_positions.func(block)}

            for position in self._possible_positions_for(
                    block, occupied_positions_by_other_blocks):
                new_block = replace(block, position=position)
                new_grid = Grid(frozenset(other_blocks | {new_block}))
                result.append(new_grid)

        return frozenset(result)

    def _possible_positions_for(
            self, block: Block, occupied_positions_by_other_blocks: Collection[Position]) -> Collection[Position]:

        positions: list[Position] = []

        if is_fixed_block(block):
            return positions

        head_position = block.pos
        head_x_position, head_y_position = head_position.x, head_position.y

        if is_HBlock(block):

            delta_xs_to_left = range(-head_x_position, 0)
            for delta_x in delta_xs_to_left:
                new_block = move_block_by_delta(block, delta_x=delta_x)
                not_possible_to_move_further_left = have_overlapp(
                    new_block, occupied_positions_by_other_blocks)

                if not_possible_to_move_further_left:
                    break

                positions += [new_block.pos]

            delta_xs_to_right = range(
                1, (GRID_SIZE - (block.size - 1) - head_x_position), 1)
            for delta_x in delta_xs_to_right:
                new_block = move_block_by_delta(block, delta_x)
                not_possible_to_move_further_right = have_overlapp(
                    new_block, occupied_positions_by_other_blocks)

                if not_possible_to_move_further_right:
                    break

                positions += [new_block.pos]

            return positions

        elif is_VBlock(block):

            delta_ys_to_top = range(-head_y_position, 0)
            for delta_y in delta_ys_to_top:
                new_block = move_block_by_delta(block, delta_y=delta_y)
                not_possible_to_move_further_up = have_overlapp(
                    new_block, occupied_positions_by_other_blocks)
                if not_possible_to_move_further_up:
                    break

                positions += [new_block.pos]

            delta_ys_to_bottom = range(
                1, (GRID_SIZE - (block.size - 1) - head_y_position), 1)
            for delta_y in delta_ys_to_bottom:

                new_block = move_block_by_delta(block, delta_y=delta_y)
                not_possible_to_move_further_down = have_overlapp(
                    new_block, occupied_positions_by_other_blocks)
                if not_possible_to_move_further_down:
                    break

                positions += [new_block.pos]

            return positions

        raise TypeError(block)


class OptimizedBlockMover():

    def derive_block_movements(self, x: Grid) -> Collection[Grid]:
        positions_occupied = self.__calculate_occupied_positions(x)
        reachable_positions = self.__calculate_positions_reachable_by_block(
            x, positions_occupied)
        possible_movements = self.__calculate_block_placements(
            reachable_positions)

        return frozenset([
            change_block_to_new_position(x, old_block, pos)
            for old_block, poses in possible_movements.items() for pos in poses
        ])

    def __calculate_occupied_positions(
            self, grid: Grid) -> Mapping[Position, bool]:
        return defaultdict(lambda: False, {
                           pos: True for block in grid for pos in occupied_positions(block)})

    def __calculate_positions_reachable_by_block(
        self,
        grid: Grid,
        positions_occupied: Mapping[Position, bool]
    ) -> Mapping[Block, Sequence[Position]]:

        return dict([
            (block,
             list(chain(
                 takewhile(
                    lambda x: positions_occupied[x] is False,
                    from_position_to_end_of_grid(tail_of_block(block),
                                                 direction=block.direction, include_it_self=False)),
                 takewhile(
                     lambda x: positions_occupied[x] is False,
                     from_position_to_start_of_grid(head_of_block(block),
                                                    direction=block.direction, include_it_self=False))
             )))
            for block in grid if is_movable_block(block)
        ])

    def __calculate_block_placements(
        self,
        new_reachable_positions: Mapping[Block, Sequence[Position]]
    ):
        possible_new_positions: MutableMapping[Block,
                                               MutableSequence[Position]] = defaultdict(list)

        for block, positions in new_reachable_positions.items():
            if is_HBlock(block):
                for position in positions:
                    assert position.x != block.pos.x

                    if position.x < block.pos.x:
                        possible_new_positions[block].append(position)

                    elif block.pos.x < position.x:
                        possible_new_positions[block].append(
                            Position(position.x - block.size + 1, position.y))

            elif is_VBlock(block):
                for position in positions:
                    assert position.y != block.pos.y

                    if position.y < block.pos.y:
                        possible_new_positions[block].append(position)

                    elif block.pos.y < position.y:
                        possible_new_positions[block].append(
                            Position(position.x, position.y - block.size + 1))

        return possible_new_positions
