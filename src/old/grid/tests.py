
from tools.block.definition import Block, MainBlock, BrownBlock, FixedBlock
from tools.direction import Direction
from tools.grid.definition import Grid
from tools.grid.utils.validator import Validator
from tools.grid.utils.placement import Placement
from tools.grid.utils.rearrangement import Rearrangement
from tools.position.definition.position import Position


def test_have_one_red_block():
    grid = Grid()
    assert Validator.have_one_red_block(grid) == False

    grid = Grid(frozenset([MainBlock(Position(0, 0))]))
    assert Validator.have_one_red_block(grid) == True

    grid = Grid(
        frozenset([
            MainBlock(Position(0, 0)),
            MainBlock(Position(1, 0))]))
    assert Validator.have_one_red_block(grid) == False


def test_is_all_blocks_inside_grid():
    grid = Grid()
    assert Validator.is_all_blocks_inside_grid(grid) == True

    grid = Grid(frozenset([MainBlock(Position(-1, 0))]))
    assert Validator.is_all_blocks_inside_grid(grid) == False

    grid = Grid(frozenset([MainBlock(Position(0, -1))]))
    assert Validator.is_all_blocks_inside_grid(grid) == False

    grid = Grid(frozenset([MainBlock(Position(6, 0))]))
    assert Validator.is_all_blocks_inside_grid(grid) == False

    grid = Grid(frozenset([MainBlock(Position(0, 6))]))
    assert Validator.is_all_blocks_inside_grid(grid) == False

    grid = Grid(frozenset([MainBlock(Position(5, 0))]))
    assert Validator.is_all_blocks_inside_grid(grid) == False

    grid = Grid(frozenset([MainBlock(Position(4, 0))]))
    assert Validator.is_all_blocks_inside_grid(grid) == True


def test_have_no_block_collision_in_Grid():
    grid = Grid()
    assert Validator.have_no_block_collision_in_Grid(grid) == True

    grid = Grid(frozenset([
        BrownBlock(size=2, direction=Direction.V,
                   position=Position(1, 0)),
        BrownBlock(size=2, direction=Direction.H,
                   position=Position(0, 1)),

    ]))
    assert Validator.have_no_block_collision_in_Grid(grid) == False


def test_rearrangement():
    grid = Grid(frozenset([MainBlock(Position(3, 0))]))

    correct_rearrangement = {
        Grid(frozenset([MainBlock(Position(x, 0))])) for x in [0, 1, 2, 4]
    }

    assert correct_rearrangement == Rearrangement.neighbour_states(grid)

    b1 = BrownBlock(
        size=2,
        direction=Direction.H,
        position=Position(0, 0)
    )

    b2 = BrownBlock(
        size=2,
        direction=Direction.V,
        position=Position(5, 0)
    )
    grid = Grid(frozenset([b1, b2]))

    correct_rearrangement = {
        BrownBlock(
            size=2,
            direction=Direction.H,
            position=Position(x, 0)) for x in [1, 2, 3]
    }

    correct_rearrangement = {Grid(frozenset(
        [BrownBlock(
            size=2,
            direction=Direction.H,
            position=Position(x, 0)),
         BrownBlock(
            size=2,
            direction=Direction.V,
            position=Position(5, 0))])) for x in [1, 2, 3]} | {

        Grid(
            frozenset([BrownBlock(
                size=2,
                direction=Direction.V,
                position=Position(5, y)),
                BrownBlock(
                size=2,
                direction=Direction.H,
                position=Position(0, 0))])) for y in [1, 2, 3, 4]

    }

    assert len(correct_rearrangement) == len(
        Rearrangement.neighbour_states(grid))

    grid = Grid(frozenset(
        [
            BrownBlock(
                size=2,
                direction=Direction.H,
                position=Position(4, 0)

            ),
            BrownBlock(
                size=2,
                direction=Direction.V,
                position=Position(5, 1)

            )]))

    correct_rearrangement = {Grid(frozenset(
        [
            BrownBlock(
                size=2,
                direction=Direction.H,
                position=Position(x, 0)

            ),
            BrownBlock(
                size=2,
                direction=Direction.V,
                position=Position(5, 1)

            )])) for x in [0, 1, 2, 3]} | {

        Grid(
            frozenset([
                BrownBlock(
                    size=2,
                    direction=Direction.H,
                    position=Position(4, 0)
                ), BrownBlock(
                    size=2,
                    direction=Direction.V,
                    position=Position(5, y)
                )
            ]))

        for y in [2, 3, 4]
    }

    assert correct_rearrangement == Rearrangement.neighbour_states(grid)


def test_placement():
    grid = Grid()
    placements = Placement.neighbour_states(grid)

    assert all([len(grid.blocks) == 1 for grid in placements])

    grid = Grid(frozenset([
        MainBlock(Position(0, 0))
    ]))

    placements = Placement.neighbour_states(grid)

    def is_only_one_red_block(grid: Grid):
        filter_res = filter(lambda block: isinstance(
            block, MainBlock), grid.blocks)
        return len(list(filter_res)) == 1

    assert any([is_only_one_red_block(grid) for grid in placements])

    grid = Grid(frozenset([
        FixedBlock(Position(0, 0))
    ]))

    redblock_placments = [Grid(frozenset([FixedBlock(Position(0, 0)), MainBlock(Position(x, y))])) for x in [
        0, 1, 2, 3, 4] for y in [3] if not (x == 0 and y == 0)]

    silverblock_placements = [Grid(frozenset([FixedBlock(Position(0, 0)), FixedBlock(Position(x, y))])) for x in [
        0, 1, 2, 3, 4, 5] for y in [0, 1, 2, 3, 4, 5] if not (x == 0 and y == 0)]

    brown_2x_horizintel_placments = [Grid(frozenset([FixedBlock(Position(0, 0)), BrownBlock(Position(x, y), direction=Direction.H, size=2)])) for x in [
        0, 1, 2, 3, 4] for y in [0, 1, 2, 3, 4, 5] if not (x == 0 and y == 0)]

    brown_2x_vertical_placments = [Grid(frozenset([FixedBlock(Position(0, 0)), BrownBlock(Position(x, y), direction=Direction.V, size=2)])) for x in [
        0, 1, 2, 3, 4, 5] for y in [0, 1, 2, 3, 4] if not (x == 0 and y == 0)]

    brown_3x_horizintel_placments = [Grid(frozenset([FixedBlock(Position(0, 0)), BrownBlock(Position(x, y), direction=Direction.H, size=3)])) for x in [
        0, 1, 2, 3] for y in [0, 1, 2, 3, 4, 5] if not (x == 0 and y == 0)]

    brown_3x_vertical_placments = [Grid(frozenset([FixedBlock(Position(0, 0)), BrownBlock(Position(x, y), direction=Direction.V, size=3)])) for x in [
        0, 1, 2, 3, 4, 5] for y in [0, 1, 2, 3] if not (x == 0 and y == 0)]

    all_possilbe_placments = frozenset(redblock_placments + silverblock_placements + brown_2x_horizintel_placments +
                                       brown_2x_vertical_placments + brown_3x_vertical_placments + brown_3x_horizintel_placments)

    assert len(all_possilbe_placments) == len(Placement.neighbour_states(grid))
