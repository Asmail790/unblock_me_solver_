from tools.common.constants import ML_ClASS_NBR
from abc import ABC, abstractmethod
from typing import Sequence, NamedTuple
from dataclasses import dataclass

from tools.common.interafaces.exceptions.general import GuiderException


@dataclass
class BoundingBox:
    clazz: ML_ClASS_NBR
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


class NextStep(NamedTuple):
    from_: BoundingBox
    to: BoundingBox
    message: str


class JavaToPythonInterFace(ABC):
    """
    Guides the user.
    Throws GuiderException when no guide can be found.

    Main purpose is to make python integration with Android easy.
    In __main__.py the method should only need take care of transformation between
    the Java equivalent of BoundingBox and NextStep. All other information else
    should be contained JavaToPythonInterFace.
    """
    @abstractmethod
    def guide(
        self,
        blockprops: Sequence[BoundingBox]
    ) -> NextStep | GuiderException: ...
