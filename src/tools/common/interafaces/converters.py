from abc import ABC, abstractmethod
from typing import TypeVar, Generic, NamedTuple
from collections.abc import Sequence
from collections.abc import Set

from tools.common.interafaces.java_to_python_interface import BoundingBox, NextStep

_GRAPH = TypeVar("_GRAPH")
_NODE = TypeVar("_NODE")
_BASIC_GRAPH = Set[tuple[_NODE, _NODE]]


class IGraphConverter(ABC, Generic[_GRAPH, _NODE]):

    @abstractmethod
    def convert_to_graph(self, x: _GRAPH) -> _BASIC_GRAPH:
        raise NotImplementedError()

    @abstractmethod
    def convert_from_graph(self, graph: _BASIC_GRAPH) -> _GRAPH:
        raise NotImplementedError


_INSTRUCTION = TypeVar("_INSTRUCTION")
BASIC_INSTRUCTION = NextStep


class InstructionConverter(ABC, Generic[_INSTRUCTION]):

    @abstractmethod
    def convert_to_basic_instruction(
            self, instruction: _INSTRUCTION) -> BASIC_INSTRUCTION:
        raise NotImplementedError()


_GRID = TypeVar("_GRID")

BLOCK = TypeVar("BLOCK")


class BASIC_GRID_OUTPUT(NamedTuple):
    grid_bouding_box: BoundingBox
    blocks_bounding_boxes: Sequence[BoundingBox]


BASIC_GRID_INPUT = Sequence[BoundingBox]


class IGridConverter(ABC, Generic[_GRAPH]):

    @abstractmethod
    def convert_to_grid(self, x: _GRID) -> BASIC_GRID_OUTPUT:
        raise NotImplementedError

    @abstractmethod
    def convert_from_grid(self, grid: BASIC_GRID_INPUT) -> _GRID:
        raise NotImplementedError()
