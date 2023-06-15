from typing import Generic, TypeVar
from collections.abc import Sequence
from abc import abstractmethod

X = TypeVar("X")


class IProducer(Generic[X]):
    """
    Produces new Grid.
    Each Grid produced is unique.
    """

    @abstractmethod
    def __call__(self) -> Sequence[X]:
        raise NotImplementedError()
