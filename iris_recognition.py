import cv2
import numpy as np

class iris_detection():
    def __init__(self, image_path):
        self._img = None
        self._img_gray = None
        self._img_RGB = None
        self._img_path = image_path
        self._pupil_center_x = None
        self._pupil_center_y = None
        self._pupil_radius = None
        self._iris_center_x = None
        self._iris_center_y = None
        self._iris_radius = None
    def load_image(self):
        self._img = cv2.imread(self._img_path)
        if type(self._img) is type(None):
            return False
        else:
            self._img_gray = cv2.cvtColor(self._img_path, cv2.COLOR_BGR2GRAY)
            self._img_RGB = cv2.cvtColor(self._img_path, cv2.COLOR_BGR2RGB)
            return True
    def iris_localization(self):
        image = cv2.GaussianBlur(self._img_gray,(15,15), 0)
        edges = cv2.Canny(image,50,100)
        pupil_edge = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,2, minDist = 70,
                                    param2=50,minRadius=20,maxRadius=60)

        pupil_edge = np.uint16(np.around(pupil_edge))
        for value in pupil_edge[0,:]:
            self._pupil_center_x = value[0]
            self._pupil_center_y = value[1]
            self._pupil_radius = value[2]

        blank = np.zeros_like(self._img_gray)
        mask_pupil = cv2.circle(blank, (self._pupil_center_x, self._pupil_center_y), self._pupil_radius, (255,255,255), -1)
        mask_pupil = cv2.bitwise_not(mask_pupil)
        image = cv2.bitwise_and(self._img_gray, mask_pupil)
        mask_background = cv2.circle(blank, (self._pupil_center_x, self._pupil_center_y), self._pupil_radius*3, (255,255,255), -1)
        image = cv2.bitwise_and(image, mask_background)

        image = cv2.equalizeHist(image)
        image = cv2.GaussianBlur(image,(15,15),0)

        # cv2.imshow('detected circles',test)
        # cv2.waitKey(0)

        edges = cv2.Canny(image, 50, 60)

        # cv2.imshow('detected circles',edges)
        # cv2.waitKey(0)

        iris_edge = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,2,minDist = 1000,
                                    param2=10,minRadius=pupil[2]+20,maxRadius=((pupil[2]*3)-10))

        iris_edge = np.uint16(np.around(iris_edge))
        for value in iris_edge[0,:]:
            self._iris_center_x = value[0]
            self._iris_center_y = value[1]
            self._iris_radius = value[2]

        # cv2.circle(image,(iris[0],iris[1]),iris[2],color = (0,255, 0), thickness = 2)
        # cv2.circle(image,(pupil[0],pupil[1]),pupil[2],color = (0,255, 0), thickness = 2)

        # cv2.imshow('detected circles',image)
        # cv2.waitKey(0)

        blank = np.zeros_like(self._img_gray)
        mask_pupil = cv2.circle(blank, (self._pupil_center_x, self._pupil_center_y), self._pupil_radius, (255,255,255), -1)
        mask_pupil = cv2.bitwise_not(mask_pupil)
        image = cv2.bitwise_and(self._img_gray, mask_pupil)
        mask_background = cv2.circle(blank, (self._iris_center_x, self._iris_center_y), self._iris_radius, (255,255,255), -1)
        image = cv2.bitwise_and(image, mask_background)

        # cv2.imshow('detected circles',result)
        # cv2.waitKey(0)

        return image
    def iris_normalization(self, image):
        theta = np.arange(0.00, np.pi * 2, 0.01) #array of columns in final image (goes to 6.28 because period is 2 pi)
        r = np.arange(0, self._iris_radius - self._pupil_radius, 1) #array of rows in final image
        final = image[self._iris_center_x - self._iris_radius : self._iris_center_x + self._iris_radius,
                self._iris_center_y - self._iris_radius : self._iris_center_y + self._iris_radius]
        # cv2.imshow('detected circles',final)
        # cv2.waitKey(0)
        cartesian_img = np.empty(shape = [self._iris_radius - self._pupil_radius,
                                          self._iris_radius*2,
                                          3]) # empty array of dimensions of final image
        m = interp1d([np.pi * 2, 0],[0, self._iris_radius*2]) #interpolate location on x axis
            # interpolation between x = values from 0 and 6.28 and y = 0 to width of final image

        # calculate all pixel values for normalized cartesian image
        for z in r:
            i = z + self._pupil_radius
            for j in theta:
                # x = rcos(theta)
                polarX = int((i * np.cos(j)) + self._iris_radius)
                # y = rsin(theta)
                polarY = int((i * np.sin(j)) + self._iris_radius)
                cartesian_img[z][int(m(j) - 1)] = final[polarY][polarX]


        # cartesian_img = cartesian_img[:][pupil[2]:iris[2]] #patch
        cartesian_img = cartesian_img.astype('uint8')
        image = np.asarray(cartesian_img)
        # cv2.imshow('detected circles', img)
        # cv2.waitKey(0)
        # plt.imshow(img)
        # plt.show()

    def image_enhancement(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)
        # enhanced_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # plt.imshow(enhanced_img)
        return image

    def feature_extraction(self):

    def performance_evaluation(self):

    def iris_recognition(self):




