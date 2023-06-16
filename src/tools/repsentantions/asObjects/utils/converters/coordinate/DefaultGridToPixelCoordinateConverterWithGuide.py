from tools.repsentantions.asObjects.definitions.positon import Position


import numpy as np
from PIL import Image


from pathlib import Path


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
