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
        localization_image = cv2.equalizeHist(self._img_gray)
        localization_image = cv2.medianBlur(localization_image,7)

        pupil_edges = cv2.Canny(localization_image,50,70)

        pupil_circle = cv2.HoughCircles(pupil_edges,cv2.HOUGH_GRADIENT,1,20,
                                    param1=50,param2=40,minRadius=10,maxRadius=70)

        iris_edges = cv2.Canny(localization_image, 70, 150)
        iris_circle = cv2.HoughCircles(iris_edges,cv2.HOUGH_GRADIENT,1,40,
                                    param2=35,minRadius=20,maxRadius=120)

        circles = np.append(pupil_circle, iris_circle, axis=0)
        circles = np.uint16(np.around(circles))

        for circle in circles:
            for value in circle:
                center_x = value[0]
                center_y = value[1]
                radius = value[2]
                cv2.circle(localization_image,(center_x,center_y),radius,color = (0,255, 0), thickness = 2)

    def iris_normalization(self):

    def image_enhancement(self):

    def feature_extraction(self):

    def performance_evaluation(self):

    def iris_recognition(self):




