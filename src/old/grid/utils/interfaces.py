import abc
from tools.block.definition import Block
from tools.grid.definition import Grid
from typing import Any, NamedTuple, Sequence
from attrs import define
from tools.position.definition import Position
import networkx as nx
from enum import Enum, auto
from typing import TypeVar, Generic

from tools.block.definition import MainBlock, HorizontalMovableBlock2X, HorizontalMovableBlock3X, VerticalMovableBlock2X, VerticalMovableBlock3X, FixedBlock
from typing import Type
from tools.constants import CL_TO_BLOCK_MAP

X = TypeVar("X")
Y = TypeVar("Y")


class IProducer(Generic[X]):
    """
    Produces new Grid.
    Each Grid produced is unique.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, '__call__') and
            callable(subclass.__call__) or
            NotImplemented)

    @abc.abstractmethod
    def __call__(self) -> Grid:
        pass


class Deriver(Generic[X, Y]):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, '__call__') and
            callable(subclass.__call__) and
            hasattr(subclass, 'description') and
            callable(subclass.description) or
            NotImplemented)

    @abc.abstractmethod
    def __call__(self, x: X) -> frozenset[X]:
        raise NotImplementedError

    @abc.abstractmethod
    def description(self, before: X, after: X) -> Y:
        raise NotImplementedError


@define(frozen=True)
class PlacementAction():
    new_block_added: Block


@define(frozen=True)
class BlockMovementAction():
    block: Block
    new_position: Position


class IBlockPlacer(Deriver[Grid, PlacementAction], metaclass=abc.ABCMeta):
    """
    Given a puzzle calculates all possible way to add one block to current puzzle.
    """


class IBlockMover(Deriver[Grid, BlockMovementAction], metaclass=abc.ABCMeta):
    """
    Given a puzzle calculates all possible way to move one block in the puzzle to new valid position.
    """


class IGraphBuilder(Generic[X, Y], metaclass=abc.ABCMeta):
    """
    Given a state X finds states that can be derived from X using function f using a Producer.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, '__call__') and
                callable(subclass.__call__) or
                NotImplemented)

    @abc.abstractmethod
    def __call__(self, start_node: Grid, deriver: Deriver[X, Y]) -> nx.DiGraph:
        raise NotImplementedError


class IPuzzleSolver(metaclass=abc.ABCMeta):
    """
    Given a puzzle can find solution for any state dervied for orginal puzzle state.
    If the puzzle is UnSolvablePuzzle rasies a UnSolvablePuzzle Error in init.

    -----------------------------
    In the final node MainBlock is positioned at (4,2).
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, '__call__') and
                callable(subclass.__call__) or
                NotImplemented)

    @abc.abstractmethod
    def __call__(self, start_node: Grid) -> list[BlockMovementAction]:
        """
        Calculates the steps need to take to solve the puzzle.
        Return the solution as a list of edges or in case
        no solution is found a CauseOfFailure object.
        """
        raise NotImplementedError


class IPuzzleSolvedChecker(metaclass=abc.ABCMeta):
    """p
    Check if the puzzle is solved.
    -----------------------------
    A puzzle is solved if MainBlock is positioned at (4,2).
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, '__call__') and
                callable(subclass.__call__) or
                NotImplemented)

    @abc.abstractmethod
    def __call__(self, puzzle: Grid) -> bool:
        raise NotImplementedError


class IPuzzleValidator(metaclass=abc.ABCMeta):
    """
    Check if the puzzle is valid.
    -----------------------------
    A valid puzzle is a puzzle where
    1) There is no block that overlap with another block.
    2) All blocks are inside the grid.
    3) There is exactly one MainBlock with position.y == 2.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, '__call__') and
                callable(subclass.__call__) or
                NotImplemented)

    @abc.abstractmethod
    def __call__(self, grid: Grid) -> bool:
        raise NotImplementedError


class IGridToPixelCoordinateTransformer():
    """
    Transform logical grid coordinate of a block to a image coordinate.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, '__call__') and
                callable(subclass.__call__) or
                NotImplemented)

    @abc.abstractmethod
    def __call__(self, position: Position) -> tuple[float, float]:
        pass


class IPixelToGridCoordinateTransformer():
    """
    Transform image coordinate of a block to a logical grid coordinate.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, '__call__') and
                callable(subclass.__call__) or
                NotImplemented)

    @abc.abstractmethod
    def __call__(self, coordinate: tuple[float, float]) -> Position:
        pass


class MLClassToPythonClassConverter():

    def __call__(self, class_: int, position: Position) -> Block:
        if issubclass(CL_TO_BLOCK_MAP[class_], Block):
            return CL_TO_BLOCK_MAP[class_](position)  # type: ignore

        raise ValueError()


class Theme(Enum):
    AUTUMN = auto()
    CHERRY_WOOD = auto()
    CHINESE_NEW_YEAR = auto()
    CHOCCY_SWEETIE = auto()
    CHRISTMAS = auto()
    EASTER = auto()
    HALLOWEEN = auto()
    MAPLE_WOOD = auto()
    OAK_WOOD = auto()
    OCEAN_BLUE = auto()
    ORIGNAL_1 = auto()
    ORIGNAL_2 = auto()
    OUR_5TH_YEAR = auto()
    PINE_WOOD = auto()
    SERENE_SIAM_SONGKRAN = auto()
    SPRING = auto()
    SUMMER = auto()
    VALENTINE = auto()
    WINTER = auto()
    WOODEN_BRIGE = auto()


@define(frozen=True)
class SampleInfo:
    theme: Theme
    grid: Grid


class IDataSetGenerator:
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, '__call__') and
                callable(subclass.__call__) or
                NotImplemented)

    @property
    @abc.abstractmethod
    def __old_samples(self) -> frozenset[SampleInfo]:
        pass

    @abc.abstractmethod
    def __save(self, new_grids: frozenset[SampleInfo]):
        pass

    def __call__(self) -> None:
        unsaved_samples = self.__new_samples - self.__old_samples
        self.__save(unsaved_samples)

    @property
    @abc.abstractmethod
    def __new_samples(self) -> frozenset[SampleInfo]:
        pass


A = TypeVar('A', bound=SampleInfo)
B = TypeVar('B', bound=SampleInfo)


class DataSetConverter(Generic[A, B]):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, '__call__') and
                callable(subclass.__call__) or
                NotImplemented)

    @abc.abstractmethod
    def __call__(self, sampleInfo: A) -> B:
        pass
