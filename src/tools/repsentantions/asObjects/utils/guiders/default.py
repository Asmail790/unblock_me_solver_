from collections.abc import Mapping
from typing import Sequence
from uuid import UUID

from tools.common.interafaces.java_to_python_interface import BoundingBox, NextStep
from tools.common.interafaces.java_to_python_interface import JavaToPythonInterFace
from tools.repsentantions.asObjects.definitions.grid import Grid
from tools.repsentantions.asObjects.utils.checkers.is_valid_checekers import PuzzleValidator
from tools.repsentantions.asObjects.utils.derivers.movers import OptimizedBlockMover
from tools.repsentantions.asObjects.utils.derivers.describers import DefaultBLockMoveDescriber

from tools.repsentantions.asObjects.utils.derivers.composers import BLockMoverComposer
from tools.repsentantions.asObjects.utils.derivers.graphbuilders import OptimizedBFSGraphBuilderWithEarlyStop
from tools.repsentantions.asObjects.utils.checkers.is_solved_checkers import OptimizedPuzzleSolvedChecker
from tools.repsentantions.asObjects.utils.solvers.defaultsolver import DefaultPuzzleSolver
from tools.repsentantions.asObjects.utils.converters.grid import DefaultGridConverter
from tools.repsentantions.asObjects.utils.converters.instruction import DefaultInstructionConverter


class DeafaultJPI(JavaToPythonInterFace):
    """
    Guides the user.
    Throws GuiderException when no guide can be found.
    """

    def __init__(self) -> None:
        self.validator = PuzzleValidator()
        self.mover = BLockMoverComposer(
            OptimizedBlockMover(),
            DefaultBLockMoveDescriber()
        )
        self.__builder = OptimizedBFSGraphBuilderWithEarlyStop()
        self.__terminal_node_checker = OptimizedPuzzleSolvedChecker()
        self.__solver = DefaultPuzzleSolver(
            deriver=self.mover,
            validator=self.validator,
            builder=self.__builder,
            terminal_node_checker=self.__terminal_node_checker
        )
        self.__grid_converter: DefaultGridConverter | None = None
        self.__next_step_converter: None | DefaultInstructionConverter = None

    def guide(self, boundingboxes: Sequence[BoundingBox]) -> NextStep:

        if self.__grid_converter is None or self.__next_step_converter is None:
            self.__grid_converter = DefaultGridConverter(boundingboxes)
            self.__next_step_converter = DefaultInstructionConverter(
                self.__grid_converter)

        grid = Grid(self.__grid_converter.convert_from_grid(boundingboxes))
        _, edges = self.__solver.solve(grid)
        step = edges[0]

        return self.__next_step_converter.convert_to_basic_instruction(step)
