from networkx import DiGraph
from collections import deque
from math import inf

from tools.common.constants import FINISH_POSITION
from tools.repsentantions.asObjects.definitions.block import MainBlock
from tools.common.interafaces.graph_builder import IGenericGraphBuilder
from tools.common.interafaces.deriver import Deriver
from tools.repsentantions.asObjects.definitions.grid import Grid
from tools.repsentantions.asObjects.utils.derivers.descriptions import MovementDescription


class OptimizedBFSGraphBuilder(
        IGenericGraphBuilder[DiGraph, Grid, MovementDescription]):
    def build_graph(
        self, start_node: Grid,
        adjacent: Deriver[Grid, MovementDescription],
        max_grids=inf
    ) -> DiGraph:

        queue: list[Grid] = [start_node]
        visited = set()
        graph = DiGraph()

        while len(queue) > 0:
            if len(graph.nodes) == max_grids:
                break

            visited.update(queue)

            new_edges = [(node, new_node)
                         for node in queue for new_node in adjacent.derive_nodes(node)]

            graph.add_edges_from(
                map(lambda
                    edge: (
                        node_from := edge[0],
                        node_to := edge[1],
                    ), new_edges))

            queue = list({new_node for _,
                          new_node in new_edges}.difference(visited))

        return graph


class OptimizedBFSGraphBuilderWithEarlyStop(
        IGenericGraphBuilder[DiGraph, Grid, MovementDescription]):
    def build_graph(self,
                    start_node: Grid,
                    adjacent: Deriver[Grid, MovementDescription],
                    max_grids=inf
                    ) -> DiGraph:
        queue: list[Grid] = [start_node]
        visited = set()
        graph = DiGraph()
        main_blocks = MainBlock(FINISH_POSITION)

        terminal_node_found = False
        while len(queue) > 0 and (not terminal_node_found):
            if len(graph.nodes) == max_grids:
                break

            visited.update(queue)
            terminal_node_found = any(
                map(lambda g: main_blocks in g, queue))

            new_edges = [(node, new_node)
                         for node in queue for new_node in adjacent.derive_nodes(node)]

            graph.add_edges_from(
                map(lambda
                    edge: (
                        node_from := edge[0],
                        node_to := edge[1]
                    ), new_edges))

            queue = list({new_node for _,
                          new_node in new_edges}.difference(visited))

        return graph


class BFSGraphBuilder(
        IGenericGraphBuilder[DiGraph, Grid, MovementDescription]):
    def build_graph(self,
                    start_node: Grid,
                    adjacent: Deriver[Grid, MovementDescription],
                    max_grids=inf
                    ) -> DiGraph:
        queue: deque[Grid] = deque([start_node])
        visited = set()
        graph = DiGraph()

        while len(queue) > 0:
            if len(graph.nodes) == max_grids:
                break

            node = queue.pop()
            visited.add(node)

            for new_node in adjacent.derive_nodes(node):

                graph.add_edge(node, new_node)
                if new_node not in visited:
                    queue.append(new_node)

        return graph
