from collections import defaultdict
from termcolor import colored

from tools.repsentantions.asObjects.definitions.grid import Grid
from tools.repsentantions.asObjects.definitions.block import H2XBlock, H3XBlock, V2XBlock, V3XBlock, FixedBlock, MainBlock
from tools.repsentantions.asObjects.utils.block.positions import occupied_positions_by_block
from tools.repsentantions.asObjects.definitions.positon import Position


_ALPHABET = [char for char in "abcdefghijklmnopqrstuvwxyz"]
_CAPITAL_APHABET = [letter.capitalize() for letter in _ALPHABET]
_DIGTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
_DRAW_WORLD_CHARS = _DIGTS + _ALPHABET + _CAPITAL_APHABET


def draw_in_terminal(puzzle: Grid) -> str:
    world: defaultdict[Position, str] = defaultdict(lambda: "□")

    for idx, block in enumerate(puzzle):
        idx = _DRAW_WORLD_CHARS[idx]
        for position in occupied_positions_by_block(block):

            match (block):
                case MainBlock(_):
                    world[position] = colored(str(idx), color="red")

                case H2XBlock(_):
                    world[position] = colored(str(idx), color="green")

                case H3XBlock(_):
                    world[position] = colored(str(idx), color="yellow")

                case V2XBlock(_):
                    world[position] = colored(str(idx), color="magenta")

                case V3XBlock(_):
                    world[position] = colored(str(idx), color="white")

                case FixedBlock(_):
                    world[position] = colored(str(idx), color="blue")

                case _:
                    raise TypeError()

    worldstr = ""
    for y in range(6):
        for x in range(6):
            worldstr += world[Position(x, y)]
        worldstr += "\n"

    return worldstr
