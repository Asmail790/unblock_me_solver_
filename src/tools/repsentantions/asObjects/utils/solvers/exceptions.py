
from tools.common.interafaces.java_to_python_interface import GuiderException


class NoSolutionExist(GuiderException):
    def __init__(self) -> None:
        super().__init__(message="There are no terminal nodes in Graph.")


class UnvalidPuzzle(GuiderException):

    def __init__(self) -> None:
        super().__init__(message="Some block are overlapping.")
