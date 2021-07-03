from typing import List
from nptyping import NDArray, UInt


x, y = 1920, 1080
Color = List[UInt[8]]
ImageThreeChannel = NDArray[(x, y, 3), UInt[8]]
ImageSingleChannel = NDArray[(x, y, 1), UInt[8]]
