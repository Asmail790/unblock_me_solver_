from abc import ABC, abstractmethod
from typing import TypeVar, Generic

X = TypeVar("X")


class IPuzzleValidator(ABC, Generic[X]):
    """
    Check if the puzzle is valid.
    -----------------------------
    A valid puzzle is a puzzle where
    1) There is no block that overlap with another block.
    2) All blocks are inside the grid.
    3) There is exactly one MainBlock with position.y == 2.
    """
    @abstractmethod
    def is_valid(self, grid: X) -> bool:
        raise NotImplementedError()


class IPuzzleSolvedChecker(ABC, Generic[X]):
    """
    Check if the puzzle is solved.
    -----------------------------
    A puzzle is solved if MainBlock is positioned at (4,2).
    """

    @abstractmethod
    def is_solved(self, grid: X) -> bool:
        raise NotImplementedError()
