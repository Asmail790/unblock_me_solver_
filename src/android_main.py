from tools.repsentantions.asObjects.utils.guiders.default import DeafaultJPI
from tools.common.interafaces.java_to_python_interface import JavaToPythonInterFace, BoundingBox
from com.example.unblockmesolver import R
from android.graphics import RectF


interface: JavaToPythonInterFace = DeafaultJPI()


def k(blocks_rects, grid_rect):
    blocks = [BoundingBox(
        clazz=x.classIndex,
        left=x.rect.left,
        right=x.rect.right,
        top=x.rect.top,
        bottom=x.rect.bottom
    ) for x in blocks_rects
    ]

    gridprops = BoundingBox(
        clazz=grid_rect.classIndex,
        left=grid_rect.rect.left,
        right=grid_rect.rect.right,
        top=grid_rect.rect.top,
        bottom=grid_rect.rect.bottom
    )
    nextStep = interface.guide(blocks + [gridprops])

    from_rect = RectF(
        nextStep.from_.left,
        nextStep.from_.left.top,
        nextStep.from_.left.right,
        nextStep.from_.left.bottom
    )
    to_rect = RectF(
        nextStep.to.left,
        nextStep.to.left.top,
        nextStep.to.left.right,
        nextStep.to.left.bottom
    )
    explation = nextStep.message

    return NextStep(from_rect, to_rect, explation)
