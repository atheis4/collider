from functools import cache
import numpy as np
import random
import cv2

from glob import glob

from utils import constants
from img_processing import (
    apply_color_to_layer,
    convert_to_3_channel,
    convert_to_black_and_white_binary,
    invert_black_and_white,
)


# COMPONENT: facial recognition data filepaths

# COMPONENT: static images on disk
random_img_path = glob("/Users/andrew/art/collider/source/version_1/horizontal/*")
num_files = len(random_img_path)


class FacialRecognition:
    CASCADE_ROOT = "/Users/andrew/code/collider/.venv/lib/python3.9/site-packages/cv2/data"
    # eye_data = root + "/haarcascade_eye.xml"
    # smile = root + "/haarcascade_smile.xml"
    # cat = root + "/haarcascade_frontalcatface.xml"

    def __init__(self):
        cascade = self.CASCADE_ROOT + "/haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(cascade)

    def identify_faces(self, frame):
        self.faces = self.classifier.detectMultiScale(
            frame, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE
        )

    def is_observer_present(self):
        return len(self.faces) > 0

    def outline_faces(self, frame):
        """Draw rectangle around faces that are in frame."""
        for x, y, w, h in self.faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), constants.Color.MAGENTA, 5)


class Camera:
    def __init__(self):
        self.capture = cv2.VideoCapture(constants.Camera.WEBCAM)
        if not self.capture.isOpened():
            # TODO: improve flow control on failure
            print("cannot open camera.")
            exit()

    def next_frame(self):
        success, frame = self.capture.read()
        if not success:
            print("could not receive frame... :(")
            # TODO: improve flow control on failure
            exit()
        return frame

    def release_capture(self):
        self.capture.release()


class QuantumObserver:
    def __init__(self):
        self.camera = Camera()
        self.classifer = FacialRecognition()

    def run(self):
        while True:
            raw_feed = self.camera.next_frame()
            frame = cv2.cvtColor(raw_feed, cv2.COLOR_BGR2GRAY)
            self.classifer.identify_faces(frame)

            # TODO: new function to encapsulate 
            frame = convert_to_black_and_white_binary(frame)
            # frame = invert_black_and_white(frame)
            frame = convert_to_3_channel(frame)
            frame = apply_color_to_layer(frame, constants.Color.RED)
            frame = np.array(frame, dtype=np.uint8)

            if self.classifer.is_observer_present():
                cv2.imshow("observer_effect", raw_feed)
            else:
                self.classifer.outline_faces(frame)
                cv2.imshow("observer_effect", frame)

            if cv2.waitKey(1) == ord('q'):
                break
        self.cleanup()

    def cleanup(self):
        self.camera.release_capture()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    quantum_observer = QuantumObserver()
    quantum_observer.run()
