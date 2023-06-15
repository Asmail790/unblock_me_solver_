from collections.abc import Collection
from typing import Protocol

from tools.common.interafaces.deriver import Deriver
from tools.repsentantions.asObjects.definitions.grid import Grid
from tools.repsentantions.asObjects.utils.derivers.descriptions import MovementDescription
from tools.repsentantions.asObjects.utils.derivers.descriptions import PlacementDescription


class BlockMovementDeriver(Protocol):
    def derive_block_movements(self, x: Grid) -> Collection[Grid]: ...


class BlockPlacerDeriver(Protocol):
    def derive_new_block_placements(self, x: Grid) -> Collection[Grid]: ...


class BlockMovementDescriber(Protocol):
    def describe_block_movement(
        self, x: Grid, y: Grid) -> MovementDescription: ...


class BlockPlacementDescriber(Protocol):
    def describe_block_placement(
        self, x: Grid, y: Grid) -> PlacementDescription: ...


class BLockMoverComposer(Deriver[Grid, MovementDescription]):

    def __init__(self,
                 deriver: BlockMovementDeriver,
                 describer: BlockMovementDescriber
                 ) -> None:
        self.__derive_nodes = deriver
        self.__describer = describer

    def derive_nodes(self, x: Grid) -> Collection[Grid]:
        return self.__derive_nodes.derive_block_movements(x)

    def describe(self, x: Grid, y: Grid) -> MovementDescription:
        return self.__describer.describe_block_movement(x, y)
