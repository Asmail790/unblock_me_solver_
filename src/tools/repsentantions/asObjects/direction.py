from enum import Enum, auto


class Direction(Enum):
    V = auto()
    H = auto()

    def short_str(self)->str:
        return f"D[{self.name}]"
