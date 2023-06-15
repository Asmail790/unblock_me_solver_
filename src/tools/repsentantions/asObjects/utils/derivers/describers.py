from tools.common.interafaces.deriver import Deriver
from tools.repsentantions.asObjects.definitions.grid import Grid
from tools.repsentantions.asObjects.utils.derivers.descriptions import MovementDescription
from tools.repsentantions.asObjects.utils.derivers.descriptions import PlacementDescription


class DefaultBLockMoveDescriber():
    def describe_block_movement(self, x: Grid, y: Grid) -> MovementDescription:
        old_block = list(x - y)[0]
        new_block = list(y - x)[0]

        return MovementDescription(
            block=old_block, new_position=new_block.pos)


class DefaultBLockPlacementDescriber():

    def describe(self, before: Grid, after: Grid) -> PlacementDescription:
        new_block_added = list((after - before))[0]
        return PlacementDescription(new_block_added=new_block_added)
