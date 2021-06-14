import numpy as np
from typing import Callable, List

import cv2

from utils import constants


def read_image_grayscale(filepath):
    # TODO: what if file doesn't exist yo?
    return cv2.imread(filepath, 0)


def _get_threshold_grayscale_value_from_func(
    img: List[List[int]], func: Callable = np.median
) -> int:
    """
    params:
        img : must be grayscale of image
        func : a numpy function that returns the statistical mean, median, etc.

    Returns the grayscale pixel value from the threshold function.
    """
    return func(func(img, axis=1))


def convert_to_black_and_white(
    img, threshold_func=_get_threshold_grayscale_value_from_func
):
    return cv2.threshold(img, threshold_func(img), 255, cv2.THRESH_BINARY)[1]


def get_mean_grayscale_value(img: List[List[int]]) -> int:
    """
    params:
        img : must be grayscale of image

    Returns the mean grayscale pixel value.
    """
    return np.mean(np.mean(img, axis=1))


def convert_to_3_channel(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def apply_colors_to_layer(img, color1, color2):
    """
    color is represented as a list of three integers between 0, 255.
    """
    return np.where(img == constants.Color.WHITE, color1, color2)


def apply_color_to_layer(img, color, layer=constants.Color.WHITE):
    """
    color is represented as a list of three integers between 0, 255.
    """
    return np.where(img == layer, color, img)


def invert_black_and_white(img):
    return cv2.bitwise_not(img)


def invert_and_apply_color_layer(img, color):
    img = invert_black_and_white(img)
    return apply_color_to_layer(img, color)


def get_output_filename(filepath):
    """
    path/from/directory/image.jpg
    """
    filename = filepath.split("/")[-1]
    return filename.split(".")[0]


def collider_transform(image, color):
    bw = convert_to_black_and_white(image)
    three = convert_to_3_channel(bw)
    return apply_color_to_layer(three, color)
