import numpy as np
from PIL import Image
from pathlib import Path

from tools.repsentantions.asObjects.definitions.positon import Position
from tools.common.constants import GRID_SIZE
from tools.common.interafaces.java_to_python_interface import BoundingBox


class PixelToGridCoordinateConverter():

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


class GridToPixelCoordinateConverterWithoutGuide():
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


class DefaultGridToPixelCoordinateConverterWithGuide():

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
