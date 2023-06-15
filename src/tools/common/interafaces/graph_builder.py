from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from math import inf

from tools.common.interafaces.deriver import Deriver
GRAPH = TypeVar("GRAPH")
NODE = TypeVar("NODE")
DESCRIPTION = TypeVar("DESCRIPTION")


class IGenericGraphBuilder(ABC, Generic[GRAPH, NODE, DESCRIPTION]):
    @abstractmethod
    def build_graph(
        self,
        start_node: NODE,
        adjacent: Deriver[NODE, DESCRIPTION],
        max_grids=inf
    ) -> GRAPH:
        raise NotImplementedError
