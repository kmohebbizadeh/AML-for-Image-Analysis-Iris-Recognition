import numpy
import cv2

class iris_detection():
    def __init__(self, image_path):
        self._img = None
        self._img_path = image_path
        self._pupil = None
        self.iris = none
    def load_image(self):
        self._img = cv2.imread(self._img_path)
        if type(self._img) is type(None):
            return False
        else:
            return True
    def iris_localization(self):

    def iris_normalization(self):

    def image_enhancement(self):

    def feature_extraction(self):

    def performance_evaluation(self):

    def iris_recognition(self):




