import cv2
import numpy as np

class iris_detection():
    def __init__(self, image_path):
        self._img = None
        self._img_gray = None
        self._img_RGB = None
        self._img_path = image_path
        self._pupil = None
        self.iris = none
    def load_image(self):
        self._img = cv2.imread(self._img_path)
        self._img_gray = cv2.cvtColor(self._img_path, cv2.COLOR_BGR2GRAY)
        self._img_RGB = cv2.cvtColor(self._img_path, cv2.COLOR_BGR2RGB)
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




