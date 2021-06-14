import enum


class Color:
    BLACK = [0, 0, 0]
    WHITE = [255, 255, 255]
    RED = [0, 0, 255]
    GREEN = [0, 255, 0]
    BLUE = [255, 0, 0]
    CYAN = list([a + b for a, b in zip(BLUE, GREEN)])
    MAGENTA = list([a + b for a, b in zip(BLUE, RED)])
    YELLOW = list([a + b for a, b in zip(RED, GREEN)])


class Orientation(enum.Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


COLORS = {
    "red": Color.RED,
    "green": Color.GREEN,
    "blue": Color.BLUE,
}


INDEX_SETS = [
    # none inverted
    [0, 2, 4],
    [0, 4, 2],
    [2, 0, 4],
    [2, 4, 0],
    [4, 0, 2],
    [4, 2, 0],
    # last inverted
    [0, 2, 5],
    [0, 4, 3],
    [2, 0, 5],
    [2, 4, 1],
    [4, 0, 3],
    [4, 2, 1],
    # middle inverted
    [0, 3, 4],
    [0, 5, 2],
    [2, 1, 4],
    [2, 5, 0],
    [4, 1, 2],
    [4, 3, 0],
    # middle and last inverted
    [0, 3, 5],
    [0, 5, 3],
    [2, 1, 5],
    [2, 5, 1],
    [4, 1, 3],
    [4, 3, 1],
    # first inverted
    [1, 2, 4],
    [1, 4, 2],
    [3, 0, 4],
    [3, 4, 0],
    [5, 0, 2],
    [5, 2, 0],
    # first and last inverted
    [1, 2, 5],
    [1, 4, 3],
    [3, 0, 5],
    [3, 4, 1],
    [5, 0, 3],
    [5, 2, 1],
    # first and middle inverted
    [1, 3, 4],
    [1, 5, 2],
    [3, 1, 4],
    [3, 5, 0],
    [5, 1, 2],
    [5, 3, 0],
    # all three inverted
    [1, 3, 5],
    [1, 5, 3],
    [3, 1, 5],
    [3, 5, 1],
    [5, 1, 3],
    [5, 3, 1],
]
