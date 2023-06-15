from tools.grid.utils import YoloDatasetGenerator
from tools.grid.definition import Grid
from tools.grid.utils.walkers import random_walker
from tools.grid.utils.placement import Placement


grids = random_walker(Grid(), max_grids=10, produce=Placement.neighbour_states)
YoloDatasetGenerator.produce_images(grids, part_of="test", factor=3)
