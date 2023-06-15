import networkx as nx
from os import listdir
from attrs import define
from typing import Type
from attrs import evolve, field
from PIL.Image import Image as PILIMAGE
from sklearn.model_selection import train_test_split
from typing import NamedTuple
import math
from pathlib import Path
from PIL import Image
import numpy as np

from tools.grid.utils.convience import draw_puzzle
from PIL.ImageDraw import ImageDraw


from pathlib import Path
import pickle
from random import randint, choice
from tqdm import tqdm
from logging import info


from tools.block.definition import Block
from tools.direction import Direction
from tools.grid.definition import Grid
from tools.grid.utils.interfaces import IBlockPlacer
from tools.grid.utils.interfaces import IBlockMover
from tools.grid.utils.interfaces import PlacementAction
from tools.grid.utils.interfaces import BlockMovementAction
from tools.grid.utils.interfaces import IPuzzleValidator
from tools.grid.utils.interfaces import IGraphBuilder
from tools.grid.utils.interfaces import IPuzzleSolver
from tools.grid.utils.interfaces import IPuzzleSolvedChecker
from tools.grid.utils.interfaces import SampleInfo
from tools.grid.utils.interfaces import IDataSetGenerator
from tools.grid.utils.interfaces import IGridToPixelCoordinateTransformer
from tools.grid.utils.interfaces import IPixelToGridCoordinateTransformer
from tools.grid.utils.interfaces import MLClassToPythonClassConverter
from tools.grid.utils.interfaces import Theme
from tools.grid.utils.interfaces import IProducer

from tools.position.definition import Position
from tools.grid.utils.convience import have_no_MainBlock
from tools.grid.utils.convience import occupied_positions as calc_occupied_positions
from tools.grid.utils.convience import move_positions_by_delta_x
from tools.grid.utils.convience import move_positions_by_delta_y
from tools.grid.utils.convience import move_block_by_delta_x
from tools.grid.utils.convience import move_block_by_delta_y


from tools.grid.utils.convience import ALL_GRID_POSITIONS
from tools.grid.utils.convience import inside_grid
from tools.grid.utils.convience import have_no_overlapp
from tools.grid.utils.convience import FINISH_POSITION
from tools.block.definition import MainBlock
from tools.block.definition import FixedBlock
from tools.block.definition import VerticalMovableBlock2X
from tools.block.definition import VerticalMovableBlock3X
from tools.block.definition import HorizontalMovableBlock2X
from tools.block.definition import HorizontalMovableBlock3X
from time import time
from collections import deque
from itertools import product


from tools.block.definition import Block
from tools.block.utils import occupied_positions
from tools.direction import Direction
from tools.grid.definition import Grid
from tools.block.utils import is_fixed_block, is_horizontal_movable_block, is_vertical_movable_block
from tools.constants import GRID_SIZE
from tools.position.definition import Position
from tools.grid.utils.convience import have_overlapp
from typing import Type
from functools import cache
from itertools import combinations
from tools.block.definition import Block, MainBlock
from tools.block.utils import occupied_positions
from tools.grid.definition import Grid
from tools.position.definition import Position
from tools.grid.utils.convience import have_overlapp
from tools.constants import GRID_SIZE, MAINBLOCK_Y_POSITION
from tools.block.utils import is_main_block
from tools.grid.utils.interfaces import Deriver

from typing import Mapping
from collections import defaultdict
from typing import MutableMapping, Mapping, Sequence, MutableSequence
from itertools import takewhile, chain
from tools.position.utils import from_position_to_end_of_grid, from_position_to_start_of_grid
from tools.block.utils import tail_of_block, head_of_block, is_movable_block

from tools.constants import BLOCK_TO_CL_MAP, CL_TO_BLOCK_MAP

from typing import TypeVar
X = TypeVar("X")
Y = TypeVar("Y")


class DefaultBlockPlacer(IBlockPlacer):
    """
    A naive BlockPlacer.
    Works by testing positioning every type of block on every free positions.
    """
    BLOCK_TYPES: tuple[
        Type[FixedBlock],
        Type[VerticalMovableBlock3X],
        Type[VerticalMovableBlock2X],
        Type[HorizontalMovableBlock2X],
        Type[HorizontalMovableBlock3X],
        Type[MainBlock]
    ] = (
        FixedBlock,
        VerticalMovableBlock3X,
        VerticalMovableBlock2X,
        HorizontalMovableBlock2X,
        HorizontalMovableBlock3X,
        MainBlock
    )

    def __call__(self, puzzle: Grid) -> frozenset[Grid]:
        occupied_positions = calc_occupied_positions(puzzle)
        unoccupied_positions = ALL_GRID_POSITIONS - occupied_positions
        possible_placements: list[Block] = []

        for unoccupied_position in unoccupied_positions:
            possible_placements += self._possible_blocks_at(
                unoccupied_position, occupied_positions)

        new_grids: set[Grid] = {Grid(
            frozenset({*puzzle.blocks, new_block})) for new_block in possible_placements}
        return frozenset(new_grids)

    def _possible_blocks_at(self, unoccupied_position: Position,
                            occupied_positions: frozenset[Position]) -> list[Block]:
        possible_blocks: list[Block] = []

        for block_type in self.BLOCK_TYPES:
            block: Block = block_type(unoccupied_position)

            if inside_grid(block) and have_no_overlapp(
                    block, occupied_positions):
                possible_blocks += [block]

        return possible_blocks

    def description(self, before: Grid, after: Grid) -> PlacementAction:
        new_block_added = list((after.blocks - before.blocks))[0]
        return PlacementAction(new_block_added=new_block_added)


class DefaultBLockMover(IBlockMover):

    def __call__(self, grid: Grid) -> frozenset[Grid]:
        result = []

        for block in grid.blocks:
            other_blocks = frozenset(
                {otherBlock for otherBlock in grid.blocks if otherBlock != block})
            occupied_positions_by_other_blocks = frozenset(
                {pos for block in other_blocks for pos in occupied_positions(block)})

            for position in self._possible_positions_for(
                    block, occupied_positions_by_other_blocks):
                new_block = evolve(block, position=position)
                new_grid = Grid(frozenset(other_blocks | {new_block}))
                result.append(new_grid)

        return frozenset(result)

    def _possible_positions_for(
            self, block: Block, occupied_positions_by_other_blocks: frozenset[Position]) -> list[Position]:

        positions: list[Position] = []

        if is_fixed_block(block):
            return positions

        head_position = block.position
        head_x_position, head_y_position = head_position.x, head_position.y

        if is_horizontal_movable_block(block):

            delta_xs_to_left = range(-head_x_position, 0)
            for delta_x in delta_xs_to_left:
                new_block = move_block_by_delta_x(block, delta_x)
                not_possible_to_move_further_left = have_overlapp(
                    new_block, occupied_positions_by_other_blocks)

                if not_possible_to_move_further_left:
                    break

                positions += [new_block.position]

            delta_xs_to_right = range(
                1, (GRID_SIZE - (block.size - 1) - head_x_position), 1)
            for delta_x in delta_xs_to_right:
                new_block = move_block_by_delta_x(block, delta_x)
                not_possible_to_move_further_right = have_overlapp(
                    new_block, occupied_positions_by_other_blocks)

                if not_possible_to_move_further_right:
                    break

                positions += [new_block.position]

            return positions

        elif is_vertical_movable_block(block):

            delta_ys_to_top = range(-head_y_position, 0)
            for delta_y in delta_ys_to_top:
                new_block = move_block_by_delta_y(block, delta_y)
                not_possible_to_move_further_up = have_overlapp(
                    new_block, occupied_positions_by_other_blocks)
                if not_possible_to_move_further_up:
                    break

                positions += [new_block.position]

            delta_ys_to_bottom = range(
                1, (GRID_SIZE - (block.size - 1) - head_y_position), 1)
            for delta_y in delta_ys_to_bottom:

                new_block = move_block_by_delta_y(block, delta_y)
                not_possible_to_move_further_down = have_overlapp(
                    new_block, occupied_positions_by_other_blocks)
                if not_possible_to_move_further_down:
                    break

                positions += [new_block.position]

            return positions

        raise TypeError(block)

    def description(self, before: Grid, after: Grid) -> BlockMovementAction:
        old_block = list((before.blocks - after.blocks))[0]
        new_block = list((after.blocks - before.blocks))[0]

        return BlockMovementAction(
            block=old_block, new_position=new_block.position)


class OptimizedBlockMover(IBlockMover):
    def __calculate_occupied_positions(
            self, grid: Grid) -> Mapping[Position, bool]:
        return defaultdict(lambda: False, {
                           pos: True for block in grid.blocks for pos in occupied_positions(block)})

    def __calculate_positions_reachable_by_block(
        self,
        grid: Grid,
        occupied_positions: Mapping[Position, bool]
    ) -> Mapping[Block, Sequence[Position]]:

        return dict([
            (block,
             list(chain(
                 takewhile(
                    lambda x: occupied_positions[x] is False,
                    from_position_to_end_of_grid(tail_of_block(block),
                                                 direction=block.direction, include_it_self=False)),
                 takewhile(
                     lambda x: occupied_positions[x] is False,
                     from_position_to_start_of_grid(head_of_block(block),
                                                    direction=block.direction, include_it_self=False))
             )))
            for block in grid.blocks if is_movable_block(block)
        ])

    def __calculate_blockplacments(
        self,
        new_reachable_positions: Mapping[Block, Sequence[Position]]
    ):
        possible_new_positions: dict[Block,
                                     MutableSequence[Position]] = defaultdict(list)

        for block, positions in new_reachable_positions.items():
            if is_horizontal_movable_block(block):
                for position in positions:
                    assert position.x != block.position.x

                    if position.x < block.position.x:
                        possible_new_positions[block].append(position)

                    elif block.position.x < position.x:
                        possible_new_positions[block].append(
                            Position(position.x - block.size + 1, position.y))

            elif is_vertical_movable_block(block):
                for position in positions:
                    assert position.y != block.position.y

                    if position.y < block.position.y:
                        possible_new_positions[block].append(position)

                    elif block.position.y < position.y:
                        possible_new_positions[block].append(
                            Position(position.x, position.y - block.size + 1))

        return possible_new_positions

    def __call__(self, grid: Grid) -> frozenset[Grid]:
        occupied_positions = self.__calculate_occupied_positions(grid)
        reachable_positions = self.__calculate_positions_reachable_by_block(
            grid, occupied_positions)
        possilbe_movemnts = self.__calculate_blockplacments(
            reachable_positions)

        return frozenset([
            Grid(grid.blocks.difference([old_block]).union(
                [new_block := type(old_block)(pos)]))
            for old_block, poses in possilbe_movemnts.items() for pos in poses
        ])

    def description(self, before: Grid, after: Grid) -> BlockMovementAction:
        old_block = list((before.blocks - after.blocks))[0]
        new_block = list((after.blocks - before.blocks))[0]

        return BlockMovementAction(
            block=old_block, new_position=new_block.position)


class PuzzleValidator(IPuzzleValidator):
    """
    A simple IPuzzleValidator.
    """

    def __call__(self, puzzle: Grid) -> bool:
        return self._have_no_block_overlap(puzzle) and self._is_all_block_inside_grid(
            puzzle) and self._have_one_MainBlock_in_valid_position(puzzle)

    def _have_no_block_overlap(self, puzzle: Grid) -> bool:
        for b1, b2 in combinations(puzzle.blocks, 2):
            if have_overlapp(b1, b2):
                return False
        return True

    def _is_all_block_inside_grid(self, puzzle: Grid) -> bool:
        for block in puzzle.blocks:
            x, y = block.position.x, block.position.y
            if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
                return False
        return True

    def _have_one_MainBlock_in_valid_position(self, puzzle: Grid) -> bool:
        main_block_counter = 0

        for block in puzzle.blocks:
            if is_main_block(block):
                main_block_counter += 1

                if block.position.y != MAINBLOCK_Y_POSITION:
                    return False

                if main_block_counter > 1:
                    return False
        return True


class OptimizedBFSGraphBuilder(IGraphBuilder[X, Y]):
    def __call__(self, start_node: X,
                 adjacent: Deriver[X, Y], max_grids=math.inf) -> nx.DiGraph:
        before = time()
        queue: list[X] = [start_node]
        visited = set()
        graph = nx.DiGraph()

        while len(queue) > 0:
            if len(graph.nodes) == max_grids:
                break

            visited.update(queue)

            new_edges = [(node, new_node)
                         for node in queue for new_node in adjacent(node)]

            graph.add_edges_from(
                map(lambda
                    edge: (
                        node_from := edge[0],
                        node_to := edge[1],
                        {"description": adjacent.description(node_from, node_to)}), new_edges))

            queue = list({new_node for _,
                          new_node in new_edges}.difference(visited))

            # for new_node in adjacent(node):
            #
            #    #graph.add_edge(node,new_node)
            #    #graph.edges[node,new_node]["description"] = adjacent.description(node,new_node)
            #    if new_node not in visited:
            #        queue.append(new_node)

        print(time() - before, len(visited))
        return graph


class OptimizedBFSGraphBuilderWithEarlyStop(IGraphBuilder[X, Y]):
    def __call__(self, start_node: X,
                 adjacent: Deriver[X, Y], max_grids=math.inf) -> nx.DiGraph:
        before = time()
        queue: list[X] = [start_node]
        visited = set()
        graph = nx.DiGraph()
        main_blocks = MainBlock(FINISH_POSITION)

        terminal_node_found = False
        while len(queue) > 0 and (not terminal_node_found):
            if len(graph.nodes) == max_grids:
                break

            visited.update(queue)
            terminal_node_found = any(
                map(lambda g: main_blocks in g.blocks, queue))

            new_edges = [(node, new_node)
                         for node in queue for new_node in adjacent(node)]

            graph.add_edges_from(
                map(lambda
                    edge: (
                        node_from := edge[0],
                        node_to := edge[1],
                        {"description": adjacent.description(node_from, node_to)}), new_edges))

            queue = list({new_node for _,
                          new_node in new_edges}.difference(visited))

        print(time() - before, len(visited))
        return graph


class BFSGraphBuilder(IGraphBuilder[X, Y]):
    def __call__(self, start_node: X,
                 adjacent: Deriver[X, Y], max_grids=math.inf) -> nx.DiGraph:
        before = time()
        queue: deque[X] = deque([start_node])
        visited = set()
        graph = nx.DiGraph()

        while len(queue) > 0:
            if len(graph.nodes) == max_grids:
                break

            node = queue.pop()
            visited.add(node)

            for new_node in adjacent(node):

                graph.add_edge(node, new_node)
                graph.edges[node, new_node]["description"] = adjacent.description(
                    node, new_node)
                if new_node not in visited:
                    queue.append(new_node)

        print(time() - before, len(visited))
        return graph


class DefaultPuzzleSolvedChecker(IPuzzleSolvedChecker):
    def __call__(self, puzzle: Grid) -> bool:

        for block in puzzle.blocks:
            if is_main_block(block):
                return block.position == FINISH_POSITION
        return False


class OptimizedPuzzleSolvedChecker(IPuzzleSolvedChecker):
    def __call__(self, puzzle: Grid) -> bool:
        return any(is_main_block(block) and block.position ==
                   FINISH_POSITION for block in puzzle.blocks)
        return any(map(lambda block: block.position ==
                   FINISH_POSITION, filter(is_main_block, puzzle.blocks)))


class BuildResult(NamedTuple):
    graph: nx.DiGraph
    terminal_nodes: list[Grid]


class NoSolutionExist(ValueError):
    Message: str = "There are no terminal nodes in Graph."

# TODO


class UnvalidPuzzle(ValueError):

    def __init__(self, blocks_overlapping: list[tuple[Block, Block]],
                 blocks_out_of_grid: list[Block], *args: object) -> None:
        # TODO fix message
        super().__init__(*args)
        self.blocks_overlapping = blocks_overlapping
        self.blocks_out_of_grid = blocks_out_of_grid


class DefaultPuzzleSolver(IPuzzleSolver):
    def __init__(self, builder: IGraphBuilder[Grid, BlockMovementAction], start_node: Grid,
                 producer: IBlockMover, validator: IPuzzleValidator, terminal_node_checker: IPuzzleSolvedChecker):
        self.terminal_node_checker = terminal_node_checker
        self.producer = producer
        self.builder = builder
        self.start_node = start_node
        self.validator = validator
        result = self._build_graph_and_find_terminal_nodes(
            start_node, self.terminal_node_checker)
        self.graph = result.graph
        self._terminal_nodes = result.terminal_nodes

    def __call__(
            self, node: Grid) -> tuple[list[Grid], list[BlockMovementAction]]:
        x = time()
        if node not in self.graph.nodes:
            self._build_graph_and_find_terminal_nodes(
                node, self.terminal_node_checker)

        path: list[Grid] = list(
            min([
                nx.shortest_path(
                    self.graph, source=node,
                    target=terminal_node,
                    method='dijkstra'
                ) for terminal_node in self._terminal_nodes], key=len)
        )

        block_movements: list[BlockMovementAction] = [self.graph.edges[(
            path[i], path[i + 1])]["description"] for i in range(len(path) - 1)]
        print("finding", time() - x)
        return path, block_movements

    def _build_graph_and_find_terminal_nodes(
            self, initial_node: Grid, terminal_node_checker: IPuzzleSolvedChecker) -> BuildResult:
        x = time()
        graph = self.builder(initial_node, self.producer)
        terminal_nodes: list[Grid] = [
            node for node in graph.nodes if terminal_node_checker(node)]

        if len(terminal_nodes) == 0:
            raise NoSolutionExist()

        print("terminal node", time() - x)
        return BuildResult(graph=graph, terminal_nodes=terminal_nodes)


class SimpleDataSetGenerator():
    def __init__(self, samples_folder: Path, max_blocks_in_grid: int = 15,
                 samples_size: int = 10000) -> None:
        self.__BlockPlacer = DefaultBlockPlacer()
        self.__samples_size = samples_size
        self.__samples_folder = samples_folder
        self.__max_blocks_in_grid = max_blocks_in_grid
        self.__old_samples, self.__starting_point = self.__load_old_samples(
            self.__samples_folder)

    def __load_old_samples(
            self, samples_folder: Path) -> tuple[set[Grid], int]:
        info("loading old grids")
        old_samples = set()
        filenames: list[int] = list()

        for file_name in tqdm(listdir(samples_folder),
                              desc="loading old grids"):
            with open(samples_folder / file_name, "rb") as file:

                old_sample = pickle.load(file)
                old_samples.add(old_sample)

                name = int(Path(file_name).stem)
                filenames.append(name)

        starting_point = max(filenames) + 1 if len(filenames) else 0

        info(f"loaded {len(old_samples)} samples")
        info(f"starting_point for new samples are {starting_point}")
        return old_samples, starting_point

    def __sample(self):
        info("sampling new grids")
        collision_with_old_samples_count = 0
        collision_with_new_samples = 0
        no_more_options_left_count = 0

        new_samples: set[Grid] = set()
        for _ in tqdm(range(self.__samples_size), desc="sampling new grids"):
            nbr_of_blocks_in_grid = randint(3, self.__max_blocks_in_grid)

            grid = Grid(frozenset())

            for _2 in range(nbr_of_blocks_in_grid + 1):

                new_grids_with_one_new_block = self.__BlockPlacer(grid)
                new_grids_not_in_new_samples = list(
                    filter(
                        lambda x: x not in new_samples,
                        new_grids_with_one_new_block))

                if len(new_grids_not_in_new_samples) == 0:
                    no_more_options_left_count += 1
                    break

                grid = choice(new_grids_not_in_new_samples)

            if grid not in self.__old_samples:
                if grid in new_samples:
                    collision_with_new_samples += 1
                else:
                    new_samples.add(grid)

            else:
                collision_with_old_samples_count += 1

        info(
            f"sampled {len(new_samples)} new grids. Target sample size is {self.__samples_size}.")
        info(
            f"collision with old samples is  {collision_with_old_samples_count}")
        info(f"collision with new samples is  {collision_with_new_samples}")
        info(f"no more options left {no_more_options_left_count}")

        return new_samples

    def __save_new_samples(self, new_samples: set[Grid]):
        info(f"saving new grids")
        for count, grid in tqdm(
                enumerate(new_samples, self.__starting_point), desc="saving new grids"):
            with open(self.__samples_folder / str(count), "wb") as file:
                pickle.dump(grid, file)

        info(f"saved {len(new_samples)}")

    def generate(self):
        new_samples = self.__sample()
        self.__save_new_samples(new_samples)

        # current block list
        # blocks = randint (1,20)
        # for nbr  in blocks
        # randint position x (1,6)
        # randint position y (1,6)
        # if red block exist then radint block without red else radint block
        # if valid block add block else resample new:
        # if invalid grid resample grid.
        # return grid


class DefaultGridToPixelCoordinateTransformer(
        IGridToPixelCoordinateTransformer):

    def __init__(self, resource: Path) -> None:
        guide = Image.open(resource)
        position_pixels = np.asarray(guide)

        position_pixels = position_pixels[:, :, 3]

        self.guide_height = position_pixels.shape[0]
        self.guide_width = position_pixels.shape[1]

        guide_pixels_y_positions, guide_pixels_x_positions = np.where(
            position_pixels == 255)
        xposes = sorted(np.unique(guide_pixels_x_positions).tolist())
        yposes = sorted(np.unique(guide_pixels_y_positions.tolist()))

        all_positions: dict[Position, tuple[float, float]] = \
            {Position(idx, idy): (x, y)
             for idx, x in enumerate(xposes)
             for idy, y in enumerate(yposes)
             }
        self._coordinate_map = all_positions

        guide.close()

    def __call__(self, position: Position) -> tuple[float, float]:
        return self._coordinate_map[position]


@define
class BoundingBox:
    clazz: int
    left: float
    right: float

    top: float
    bottom: float

    def width(self):
        return self.right - self.left

    def height(self):
        return self.bottom - self.top

    def topleft(self):
        return (self.left, self.top)


class ConvertImageToGrid():

    def __init__(
            self, blockproperties: Sequence[BoundingBox], gridInfo: BoundingBox) -> None:
        self.__converter = DefaultPixelToGridCoordinateTransformer()
        self.__converter_from = GridToPixelCoordinateTransformerWithOutGuide()
        self.__gridInfo = gridInfo
        self.__img_block_to_logical_block: dict[Block, BoundingBox] = {}

        for blockproperty in blockproperties:
            position = self.__converter(gridInfo, blockproperty.topleft())

            block = CL_TO_BLOCK_MAP[blockproperty.clazz](position)

            self.__img_block_to_logical_block[block] = blockproperty

    def to_grid(self) -> Grid:
        return Grid(frozenset(self.__img_block_to_logical_block.keys()))

    def image_position_for_block(
            self, block: Block, new_position: Position) -> BoundingBox:

        block_img = self.__img_block_to_logical_block[block]
        width, height, ml_class = block_img.width(), block_img.height(), block_img.clazz
        new_x_start, new_y_start = self.__converter_from(
            self.__gridInfo, new_position)

        new_block_img_props = BoundingBox(
            ml_class,
            new_x_start,
            new_x_start + width,
            new_y_start,
            new_y_start + height)
        return new_block_img_props


@define(frozen=True)
class YoloLabel():
    clazz: int
    x_center: float
    y_center: float
    w: float
    h: float

    def __str__(self):
        return "{} {:.5f} {:.5f} {:.5f} {:.5f}".format(
            self.clazz, self.x_center, self.y_center, self.w, self.h)


@define(frozen=True)
class YoloSample:
    labels: list[YoloLabel]
    image: PILIMAGE

    def __str__(self) -> str:

        return "\n".join([str(x) for x in self.labels])


class ToYOLOV5():
    def __init__(self,
                 source_samples_folder: Path,
                 destination_labels_folder: Path,
                 destination_image_folder: Path,
                 theme_blocks_folder_path: Path,
                 guide_position_img_path: Path

                 ):
        self.__source_samples_folder = source_samples_folder
        self.__coordinateTransformer = DefaultGridToPixelCoordinateTransformer(
            guide_position_img_path)

        self.__destination_image_folder = destination_image_folder
        self.__destination_labels_folder = destination_labels_folder
        self.__background_image = Image.open(
            theme_blocks_folder_path / "background.png")

        _3XV_img = Image.open(theme_blocks_folder_path / "3XV.png")
        _2XV_img = Image.open(theme_blocks_folder_path / "2XV.png")
        _3XH_img = Image.open(theme_blocks_folder_path / "3XH.png")
        _2XH_img = Image.open(theme_blocks_folder_path / "2XH.png")
        _MainBlock_img = Image.open(theme_blocks_folder_path / "MainBlock.png")
        _FixedBlock_img = Image.open(
            theme_blocks_folder_path / "FixedBlock.png")

        self.__block_class_to_img: dict[Type[Block], PILIMAGE] = {
            MainBlock: _MainBlock_img,
            FixedBlock: _FixedBlock_img,
            HorizontalMovableBlock2X: _2XH_img,
            HorizontalMovableBlock3X: _3XH_img,
            VerticalMovableBlock2X: _2XV_img,
            VerticalMovableBlock3X: _3XV_img
        }

    def __generate_sample(self, grid: Grid) -> YoloSample:
        block_labels = list()

        width, height = self.__background_image.size
        new_hight = 640
        aspect = new_hight / height

        new_width = aspect * width
        new_size = int(new_width), int(new_hight)
        grid_image = self.__background_image.copy()

        for block in grid.blocks:
            x, y = self.__coordinateTransformer(block.position)
            block_img = self.__block_class_to_img[block.__class__]
            block_width, block_height = self.__block_class_to_img[block.__class__].size

            block_position_in_image = (int(x), int(y))
            grid_image.paste(block_img, block_position_in_image, block_img)

            center_xn = (x + block_width / 2) / \
                self.__coordinateTransformer.guide_width
            center_yn = (y + block_height / 2) / \
                self.__coordinateTransformer.guide_height
            widthn = block_width / self.__coordinateTransformer.guide_width
            heightn = block_height / self.__coordinateTransformer.guide_height

            data = YoloLabel(
                clazz=BLOCK_TO_CL_MAP[block.__class__],
                x_center=center_xn,
                y_center=center_yn,
                w=widthn,
                h=heightn
            )
            block_labels.append(data)

            # drawer = ImageDraw(grid_image)
            # left,right = np.array([ center_xn - widthn/2,center_xn + widthn/2])*self.__coordinateTransformer.guide_width
            # top,bottom  = np.array([center_yn - heightn/2, center_yn + heightn/2])*self.__coordinateTransformer.guide_height
            # drawer.rectangle(((left,top),(right,bottom)),width=2)

        return YoloSample(block_labels, image=grid_image.resize(new_size))

    def generate(self):
        filenames = listdir(self.__source_samples_folder)
        validation_and_train, test = train_test_split(filenames, test_size=0.1)
        train, validation = train_test_split(
            validation_and_train, test_size=0.2)

        for sample_set, folder_name in [
                (train, "train"), (validation, "val"), (test, "test")]:

            for filename in tqdm(sample_set, desc=f"Creating {folder_name}"):
                file_path: Path = self.__source_samples_folder / filename
                stem = file_path.stem
                sample = None

                with open(file_path, "rb") as f:
                    grid = pickle.load(f)
                    sample = self.__generate_sample(grid)

                label_path = self.__destination_labels_folder / \
                    folder_name / (stem + ".txt")

                with open(label_path, "w", encoding="utf8") as f:
                    f.write(str(sample))

                image_path = self.__destination_image_folder / \
                    folder_name / (stem + ".png")
                sample.image.save(image_path)


class GridToPixelCoordinateTransformerWithOutGuide(
        IGridToPixelCoordinateTransformer):
    def __call__(self,
                 gridImageProperties: BoundingBox,
                 position: Position
                 ) -> tuple[float, float]:

        grid_x_start, grid_x_end = gridImageProperties.left, gridImageProperties.right
        grid_y_start, grid_y_end = gridImageProperties.top, gridImageProperties.bottom

        step_size_x = (grid_x_end - grid_x_start) / GRID_SIZE
        step_size_y = (grid_y_end - grid_y_start) / GRID_SIZE

        intervals_x = [
            grid_x_start +
            step_size_x *
            step for step in range(GRID_SIZE)]
        intervals_y = [
            grid_y_start +
            step_size_y *
            step for step in range(GRID_SIZE)]

        x_pixel_position = intervals_x[position.x]
        y_pixel_position = intervals_y[position.y]

        return x_pixel_position, y_pixel_position


class DefaultPixelToGridCoordinateTransformer(
        IPixelToGridCoordinateTransformer):

    def __call__(self,
                 gridImageProperties: BoundingBox,
                 coordinate: tuple[float, float]
                 ) -> Position:

        x, y = coordinate
        grid_x_start, grid_x_end = gridImageProperties.left, gridImageProperties.right
        grid_y_start, grid_y_end = gridImageProperties.top, gridImageProperties.bottom

        step_size_x = (grid_x_end - grid_x_start) / GRID_SIZE
        step_size_y = (grid_y_end - grid_y_start) / GRID_SIZE

        intervals_x = [
            grid_x_start +
            step_size_x *
            step for step in range(GRID_SIZE)]
        intervals_y = [
            grid_y_start +
            step_size_y *
            step for step in range(GRID_SIZE)]

        return Position(
            self.__to_logical_position(intervals_x, x),
            self.__to_logical_position(intervals_y, y)
        )

    def __to_logical_position(
            self, interval: list[float], coordinate: float) -> int:
        options = enumerate(
            map(lambda position: abs(position - coordinate), interval))
        return min(options, key=lambda x: x[1])[0]


@define
class NextStep_():
    from_: BoundingBox
    to_: BoundingBox
    friendly_message: str


class GuiderException(Exception):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(kwargs["message"])
