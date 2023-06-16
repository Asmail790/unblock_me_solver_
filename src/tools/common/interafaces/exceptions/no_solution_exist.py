from tools.common.interafaces.exceptions.general import GuiderException


class NoSolutionExistException(GuiderException):
    def __init__(self) -> None:
        super().__init__("There are no terminal nodes in Graph.")
