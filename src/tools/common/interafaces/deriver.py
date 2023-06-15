from typing import TypeVar, Generic
from collections.abc import Collection
from abc import abstractmethod, ABC
NODE = TypeVar("NODE")
DESCRIPTION = TypeVar("DESCRIPTION")


class Deriver(ABC, Generic[NODE, DESCRIPTION]):
    """
    Given a Node calculates some or all possible nodes that can be dervied from node X according to a rule.

    Method describe should describe the how successor y was obtained from x.
    -----------------------------
    For example
    Given a puzzle calculates all possible way to add one block to current puzzle.
    Given a puzzle calculates all possible way to move one block in the puzzle to new valid position.
    """

    @abstractmethod
    def derive_nodes(self, x: NODE) -> Collection[NODE]:
        raise NotImplementedError()

    @abstractmethod
    def describe(self, x: NODE, y: NODE) -> DESCRIPTION:
        raise NotImplementedError()
