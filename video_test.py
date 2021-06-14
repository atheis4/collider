from functools import cache
import numpy as np
import random
import cv2

from glob import glob

from utils import constants
from img_processing import (
    collider_transform,
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
cat = root + "/haarcascade_frontalcatface.xml"

random_img_path = glob("/Users/andrew/art/collider/source/version_1/horizontal/*")
num_files = len(random_img_path)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cascade)

if not cap.isOpened():
    print("cannot open camera.")
    exit()
while True:
    # Capture frame-by-frame
    success, frame = cap.read()
    # if frame is read correctly ret is True
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        frame, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE
    )
    frame = convert_to_black_and_white(frame)
    frame = invert_black_and_white(frame)
    frame = convert_to_3_channel(frame)

    # face_observing = len(faces) == 0

    frame = np.array(frame, dtype=np.uint8)
    # Display the resulting frame
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), constants.Color.MAGENTA, 5)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
