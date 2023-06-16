from tools.common.interafaces.converters import BASIC_GRID_OUTPUT, IGridConverter, BASIC_GRID_INPUT
from tools.repsentantions.asObjects.definitions.grid import Grid
from tools.common.interafaces.java_to_python_interface import BoundingBox
from tools.common.constants import ML_STR_TO_ML_NBR, ML_NBR_TO_STR, ML_ClASS_NBR
from tools.repsentantions.asObjects.utils.converters.coordinate.PixelToGridCoordinateConverter import PixelToGridCoordinateConverter
from tools.repsentantions.asObjects.utils.converters.block import ML_STR_TO_PY_BLOCK_OBJECT, PY_BLOCK_OBJECT_TO_ML_STR
from tools.repsentantions.asObjects.utils.converters.coordinate.GridToPixelCoordinateConverterWithoutGuide import GridToPixelCoordinateConverterWithoutGuide
from tools.repsentantions.asObjects.definitions.block import Block
from collections.abc import Collection, Sequence


class DefaultGridConverter(IGridConverter[Sequence[Block]]):

    def __init__(self, bounding: Collection[BoundingBox]) -> None:
        blocks = filter(
            lambda x: ML_NBR_TO_STR[x.clazz] != 'Grid',
            bounding
        )
        self.__grid_props = list(filter(
            lambda x: ML_NBR_TO_STR[x.clazz] == 'Grid',
            bounding
        ))[0]

        width_and_hight = {ML_STR_TO_PY_BLOCK_OBJECT[ML_NBR_TO_STR[block.clazz]]: (
            block.width(), block.height()) for block in blocks
        }

        self.__block_props = width_and_hight

        self.__to_bounding_box = GridToPixelCoordinateConverterWithoutGuide()
        self.__to_grid = PixelToGridCoordinateConverter()

    def convert_from_grid(
            self, bounding_boxes: BASIC_GRID_INPUT) -> Sequence[Block]:

        def convert(block_props: BoundingBox):
            x = block_props.left
            y = block_props.top
            clazz = block_props.clazz

            image_pos = (x, y)
            logical_pos = self.__to_grid(self.__grid_props, image_pos)
            py_block_clazz = ML_STR_TO_PY_BLOCK_OBJECT[ML_NBR_TO_STR[clazz]]
            return py_block_clazz(logical_pos)

        bounding_boxes = [
            box for box in bounding_boxes if ML_NBR_TO_STR[box.clazz] != 'Grid']
        logical_grid = [convert(box) for box in bounding_boxes]

        return logical_grid

    def convert_to_grid(self, blocks: Sequence[Block]) -> BASIC_GRID_OUTPUT:
        def convert(block: Block):
            info = self.__block_props[type(block)]
            width = info[0]
            height = info[1]
            image_pos = self.__to_bounding_box(self.__grid_props, block.pos)
            x = image_pos[0]
            y = image_pos[1]

            ml_class_str = PY_BLOCK_OBJECT_TO_ML_STR[type(block)]
            ml_class_nbr = ML_STR_TO_ML_NBR[ml_class_str]

            return BoundingBox(ml_class_nbr, x, y, x + width, y + height)

        blocks_bounding_boxes = [convert(block) for block in blocks]
        return BASIC_GRID_OUTPUT(
            self.__grid_props,
            blocks_bounding_boxes
        )
