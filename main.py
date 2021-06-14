from collections import defaultdict
import numpy as np

import cv2

from utils import constants
from img_processing import (
    invert_black_and_white,
    read_image_grayscale,
    convert_to_black_and_white,
    convert_to_3_channel,
    apply_color_layer,
)


source_img_path = "/Users/andrew/art/collider/source/version_4"
orientation = constants.Orientation.HORIZONTAL
additive_images = [
    "computer_3_a.jpg",
    "computer_3_b.jpg",
    "computer_3_c.jpg",
]
zero, one, two = [x for x in additive_images]
subtraction_images = [
    'negative_computer_3_b.jpg'
]

OVERRIDE_THRESHOLD = True


def default_threshold(arg):
    return 127


def get_image_filepath(image):
    return f"{source_img_path}/{orientation.value}/{image}"


# iterate list, apply all three color layers to each image, construct unique final piece
img_to_layer = defaultdict(list)
for image in additive_images:
    filepath = get_image_filepath(image)
    img = read_image_grayscale(filepath)
    b_w = convert_to_black_and_white(img)
    inv_b_w = invert_black_and_white(b_w)
    three_channel = convert_to_3_channel(b_w)
    inv_three_channel = convert_to_3_channel(inv_b_w)
    for color, value in constants.COLORS.items():
        current_layer = apply_color_layer(three_channel, value)
        img_to_layer[image].append(current_layer)
        current_layer = apply_color_layer(inv_three_channel, value)
        img_to_layer[image].append(current_layer)

filepath = get_image_filepath(subtraction_images[0])
to_subtract = read_image_grayscale(filepath)
to_subtract = convert_to_black_and_white(to_subtract, default_threshold)
# to_subtract = invert_black_and_white(to_subtract)
to_subtract = convert_to_3_channel(to_subtract)
to_subtract = apply_color_layer(to_subtract, constants.Color.WHITE)

i = 48 * 0
for idx in constants.INDEX_SETS:
    piece = img_to_layer[zero][idx[0]]
    piece += img_to_layer[one][idx[1]]
    piece += img_to_layer[two][idx[2]]
    piece = cv2.add(piece, to_subtract)
    cv2.imwrite(f"results/6_11_2021_test_{i}.jpg", piece)
    i += 1
