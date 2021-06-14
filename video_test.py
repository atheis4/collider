from functools import cache
import numpy as np
import random
import cv2

from glob import glob

from utils import constants
from img_processing import (
    invert_black_and_white,
    read_image_grayscale,
    convert_to_black_and_white,
    convert_to_3_channel,
    apply_color_to_layer,
    apply_colors_to_layer,
)


# facial recognition data filepaths
root = "/Users/andrew/code/collider/.venv/lib/python3.9/site-packages/cv2/data"
cascade = root + "/haarcascade_frontalface_default.xml"
eye_data = root + "/haarcascade_eye.xml"
smile = root + "/haarcascade_smile.xml"

random_img_path = glob("/Users/andrew/art/collider/source/version_1/horizontal/*")
num_files = len(random_img_path)

cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier(cascade)

cached_images = set()


global_random_index = random.randint(0, num_files - 1)
global_random_filepath = random_img_path[global_random_index]

if not cap.isOpened():
    print("cannot open camera.")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # bw = convert_to_black_and_white(gray)
    # # bw = invert_black_and_white(bw)
    # three = convert_to_3_channel(bw)
    # blue = apply_color_to_layer(three, constants.Color.BLUE)
    # invert = invert_black_and_white(bw)
    # invert = convert_to_3_channel(invert)
    # invert = cv2.flip(invert, 1)
    # red = apply_color_to_layer(invert, constants.Color.RED)
    # img = cv2.add(blue, red)
    # # 1920 x 1080
    # GENERIC_SAD_KERMIT = cv2.imread("/Users/andrew/Desktop/sad_kermie.jpg")
    # sad_kermie = read_image_grayscale("/Users/andrew/Desktop/sad_kermie.jpg")
    # sad_kermie = convert_to_black_and_white(sad_kermie)
    # sad_kermie = convert_to_3_channel(sad_kermie)
    # sad_kermie = apply_color_to_layer(sad_kermie, constants.Color.GREEN)
    # img = cv2.add(img, sad_kermie)

    face_observing = len(faces) == 0
    
    random_index = random.randint(0, num_files - 1)
    random_filepath = random_img_path[random_index]
    # three = np.array(img, dtype=np.uint8)
    # Display the resulting frame
    if not face_observing:
        rand_image = cv2.imread(random_filepath)
        cv2.imshow("frame", rand_image)
    else:
        fixed_image = cv2.imread(global_random_filepath)
        while global_random_filepath in cached_images:
            if len(cached_images) == num_files:
                cached_images = set()
            global_random_index = random.randint(0, num_files - 1)
            random_filepath = random_img_path[global_random_index]
            fixed_image = cv2.imread(random_filepath)
        cached_images.add(random_img_path[global_random_index])
        cv2.imshow("frame", fixed_image)
    if cv2.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
