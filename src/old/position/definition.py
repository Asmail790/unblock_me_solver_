from attrs import define, evolve


@define(frozen=True)
class Position:
    x: int
    y: int

    def short_str(self):
        return f"P(x={self.x},y={self.y})"

    def to_end_of_grid_vertically(self):
        return [Position(self.x, i) for i in range(self.y, 6)]

    def to_end_of_grid_horinztally(self):
        return [Position(i, self.y) for i in range(self.x, 6)]

    def to_start_of_grid_vertically(self):
        return [Position(self.x, i) for i in range(self.y, 0, -1)]

    def to_start_of_grid_horinztally(self):
        return [Position(i, self.y) for i in range(self.x, 0, -1)]
