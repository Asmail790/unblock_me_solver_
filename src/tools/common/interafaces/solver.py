from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeVar, Generic


X = TypeVar("X")
Y = TypeVar("Y")


class IPuzzleSolver(ABC, Generic[X, Y]):
    """
    Given a puzzle can find solution for any state derived for original puzzle state.
    If the puzzle is UnSolvablePuzzle raises a UnSolvablePuzzle Error in init.

    -----------------------------
    In the final node MainBlock is positioned at (4,2).
    """

    @abstractmethod
    def solve(self, start_node: X) -> Sequence[Y]:
        """
        Calculates the steps need to take to solve the puzzle.
        Return the solution as a list of edges or in case
        no solution is found a CauseOfFailure object.
        """
        raise NotImplementedError()
