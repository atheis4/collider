import numpy as np
import random

import cv2

from img_processing import (
    apply_color_to_layer,
    convert_to_3_channel,
    convert_to_black_and_white_binary,
    invert_black_and_white,
)
from utils import constants


class FacialRecognition:
    CASCADE_ROOT = (
        "/Users/andrew/code/collider/.venv/lib/python3.9/site-packages/cv2/data"
    )
    # eye_data = root + "/haarcascade_eye.xml"
    # smile = root + "/haarcascade_smile.xml"
    # cat = root + "/haarcascade_frontalcatface.xml"

    def __init__(self):
        cascade = self.CASCADE_ROOT + "/haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(cascade)
        self.faces = None

    def identify_faces(self, frame):
        # TODO: optimization?
        self.faces = self.classifier.detectMultiScale(
            frame, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE
        )

    def is_observer_present(self):
        return len(self.faces) > 0

    def outline_faces(self, frame):
        """Draw rectangle around faces that are in frame."""
        for x, y, w, h in self.faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), constants.Color.WHITE, 5)


class Camera:
    def __init__(self):
        self.frames = []
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
        transformed_frame = self.transform_frame(frame)
        self.add_new_frame(transformed_frame)
        return frame

    def transform_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = convert_to_black_and_white_binary(frame)
        if random.random() < 0.5:
            frame = invert_black_and_white(frame)
        frame = convert_to_3_channel(frame)
        return frame

    def release_capture(self):
        self.capture.release()

    def add_new_frame(self, frame):
        while len(self.frames) < 3:
            self.frames.append(frame)    
        self.frames.pop(0)
        self.frames.append(frame)

    def get_frames(self):
        return self.frames


class QuantumObserver:
    def __init__(self):
        self.camera = Camera()
        self.classifer = FacialRecognition()

    def run(self):
        while True:
            raw_feed = self.camera.next_frame()
            frame = cv2.cvtColor(raw_feed, cv2.COLOR_BGR2GRAY)
            # TODO: optimize based on state of raw image?
            self.classifer.identify_faces(frame)

            if self.classifer.is_observer_present():
                # self.classifer.outline_faces(raw_feed)
                cv2.imshow("OBSERVED", raw_feed)
            else:
                frames = self.apply_quantum(self.camera.get_frames())
                frame = self.add_frames_together(frames)
                cv2.imshow("OBSERVED", frame)

            if cv2.waitKey(1) == ord("q"):
                break
        self.cleanup()

    def apply_quantum(self, frames):
        frames = [
            apply_color_to_layer(frames[i], color)
            for i, color in enumerate(constants.PRIMARY_COLORS.values())
            if i < len(frames)
        ]
        frames = [np.array(frame, dtype=np.uint8) for frame in frames]
        return frames

    def add_frames_together(self, frames):
        result = frames[0]
        for _, frame in enumerate(frames, 1):
            result += frame
        return result

    def cleanup(self):
        self.camera.release_capture()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    quantum_observer = QuantumObserver()
    quantum_observer.run()
