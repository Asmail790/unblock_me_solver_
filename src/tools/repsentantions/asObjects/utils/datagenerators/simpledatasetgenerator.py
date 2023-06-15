import pickle
from os import listdir
from logging import info
from pathlib import Path
from tqdm import tqdm
from random import randint, choice
from tools.repsentantions.asObjects.definitions.grid import Grid
from tools.repsentantions.asObjects.utils.derivers.placers import DefaultBlockPlacer

# TODO make interface for label generator and another for image generator.


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
