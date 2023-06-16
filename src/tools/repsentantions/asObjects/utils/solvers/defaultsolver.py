from networkx import DiGraph
import networkx as nx
from typing import Optional, NamedTuple
from collections.abc import Sequence

from tools.common.interafaces.graph_builder import IGenericGraphBuilder
from tools.common.interafaces.checkers import IPuzzleSolvedChecker, IPuzzleValidator
from tools.common.interafaces.deriver import Deriver
from tools.common.interafaces.solver import IPuzzleSolver
from tools.common.interafaces.exceptions.no_solution_exist import NoSolutionExistException
from tools.repsentantions.asObjects.utils.derivers.descriptions import MovementDescription
from tools.repsentantions.asObjects.definitions.grid import Grid


STEPS = tuple[Sequence[Grid], Sequence[MovementDescription]]


class BuildResult(NamedTuple):
    graph: DiGraph
    terminal_nodes: Sequence[Grid]


class DefaultPuzzleSolver(IPuzzleSolver[Grid, STEPS]):
    def __init__(self, builder: IGenericGraphBuilder[DiGraph, Grid, MovementDescription],
                 deriver: Deriver[Grid, MovementDescription],
                 validator: IPuzzleValidator[Grid],
                 terminal_node_checker: IPuzzleSolvedChecker[Grid]
                 ):
        self.terminal_node_checker = terminal_node_checker
        self.__deriver = deriver
        self.__builder = builder
        self.__validator = validator
        self.__graph: Optional[DiGraph] = None
        self.__terminal_nodes: Optional[Sequence[Grid]] = None
        self.__path: Optional[Sequence[Grid]] = None

    def solve(self, node: Grid) -> STEPS:
        exceptions = self.__validator.as_exceptions(node)
        if exceptions is not None:
            raise exceptions

        if self.__path is not None and node in self.__path:
            start_index = self.__path.index(node)
            block_movements: Sequence[MovementDescription] = [
                self.__deriver.describe(self.__path[i], self.__path[i + 1]) for i in range(start_index, len(self.__path) - 1)]

            return self.__path[start_index:-1], block_movements

        if self.__terminal_nodes is None or self.__graph is None or node not in self.__graph.nodes:
            result = self._build_graph_and_find_terminal_nodes(
                node, self.terminal_node_checker)
            self.__graph = result.graph
            self._terminal_nodes = result.terminal_nodes

        path: list[Grid] = list(
            min([
                nx.shortest_path(
                    self.__graph, source=node,
                    target=terminal_node,
                    method='dijkstra'
                ) for terminal_node in self._terminal_nodes], key=len)
        )
        self.__path = path

        block_movements: Sequence[MovementDescription] = [self.__deriver.describe(
            self.__path[i], self.__path[i + 1]) for i in range(len(path) - 1)]
        return path, block_movements

    def _build_graph_and_find_terminal_nodes(
            self, initial_node: Grid, terminal_node_checker: IPuzzleSolvedChecker) -> BuildResult:
        graph = self.__builder.build_graph(initial_node, self.__deriver)
        terminal_nodes: list[Grid] = [
            node for node in graph.nodes if terminal_node_checker.is_solved(node)]

        if len(terminal_nodes) == 0:
            raise NoSolutionExistException()

        return BuildResult(graph=graph, terminal_nodes=terminal_nodes)
