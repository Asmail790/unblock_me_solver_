from attr import define
from attrs import field, define
from tools.block.definition import Block
from tools.constants import GRID_SIZE


@define(frozen=True)
class Grid():
    blocks: frozenset[Block] = field(factory=frozenset, converter=frozenset)
    size: int = field(init=False, default=GRID_SIZE)

    def short_str(self):
        return {block.__short_str() for block in list(self.blocks)}
