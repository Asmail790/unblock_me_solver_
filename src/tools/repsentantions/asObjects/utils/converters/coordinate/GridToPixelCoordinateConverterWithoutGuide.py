from tools.common.constants import GRID_SIZE
from tools.common.interafaces.java_to_python_interface import BoundingBox
from tools.repsentantions.asObjects.definitions.positon import Position


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
