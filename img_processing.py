import os
import numpy as np
from typing import Callable, List

import cv2

from utils import constants, typing


def read_image_as_grayscale(filepath: str) -> typing.ImageSingleChannel:
    """
    Read the image located at filepath and return as an ndarray with a single gray channel.

    params:
        filepath (str) : the path to the source image.

    Raises an IOError if the file cannot be found.
    """
    if not os.path.exists(filepath):
        raise IOError(f"filepath: {filepath} not found.")
    return cv2.imread(filepath, 0)


def _get_threshold_grayscale_value_from_func(
    image: typing.ImageSingleChannel, func: Callable = np.median
) -> np.uint8:
    """
    Generate a threshold gray value for the ndarray using the function provided.

    params:
        image (typing.ImageSingleChannel) : must be grayscale of image
        func (Callable) : a function that accepts a multi-dimensional np array and collapses
            it into a single integer value.

    Returns an integer representing the threshold value that will bissect the
    """
    return func(func(image, axis=1))


def convert_to_black_and_white_binary(
    image: typing.ImageSingleChannel,
    threshold_func: Callable = _get_threshold_grayscale_value_from_func,
) -> typing.ImageSingleChannel:
    """
    Takes a single channel grayscale image and generates a binary of either black or 
    white pixels divided by the result of the threshold function.
    """
    return cv2.threshold(image, threshold_func(image), 255, cv2.THRESH_BINARY)[1]


def convert_to_3_channel(image: typing.ImageSingleChannel) -> typing.ImageThreeChannel:
    """
    Convert a single channel image (grayscale) into a three-channel image.

    Must be run before adding color back into an image or frame.
    """
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return np.array(image, dtype=np.uint8)


def apply_color_to_layer(
    image: typing.ImageThreeChannel, color: typing.Color, layer=constants.Color.WHITE
) -> typing.ImageThreeChannel:
    """
    Replaces all pixels matching the value of layer with the input color. All other 
    pixels remain unchanged.
    """
    return np.where(image == layer, color, image)


def apply_colors_to_layer(
    image: typing.ImageThreeChannel, color1: typing.Color, color2: typing.Color
) -> typing.ImageThreeChannel:
    """
    Replaces all pixels matching the value of layer with the first color parameter.
    All other pixels are set to the second color parameter.
    """
    return np.where(image == constants.Color.WHITE, color1, color2)


def invert_black_and_white(
    image: typing.ImageSingleChannel,
) -> typing.ImageSingleChannel:
    """
    Invert the input image. Black becomes white, white becomes black.
    """
    return cv2.bitwise_not(image)


def invert_and_apply_color_layer(
    image: typing.ImageSingleChannel, color: typing.Color
) -> typing.ImageThreeChannel:
    image = invert_black_and_white(image)
    image = convert_to_3_channel(image)
    return apply_color_to_layer(image, color)


def get_output_filename(filepath: str) -> str:
    """
    path/from/directory/image.jpg
    """
    filename = filepath.split("/")[-1]
    return filename.split(".")[0]


def collider_processing(filepath: str) -> typing.ImageSingleChannel:
    image = read_image_as_grayscale(filepath)
    return convert_to_black_and_white_binary(image)


def collider_transform(
    image: typing.ImageSingleChannel, color: typing.Color
) -> typing.ImageThreeChannel:
    image = convert_to_black_and_white_binary(image)
    image = convert_to_3_channel(image)
    image = apply_color_to_layer(image, color)
    return np.array(image, dtype=np.uint8)
