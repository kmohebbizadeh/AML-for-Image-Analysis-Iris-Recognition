import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings

warnings.filterwarnings("ignore")


class iris_detection:
    def __init__(self, image_path):
        self._img = None
        self._img_og = None
        self._img_path = image_path
        self._img_height = None
        self._img_width = None
        self._pupil_center_x = None
        self._pupil_center_y = None
        self._pupil_radius = None
        self._iris_center_x = None
        self._iris_center_y = None
        self._iris_radius = None

    def load_image(self):
        image = cv2.imread(self._img_path)
        if type(image) is type(None):
            return False
        else:
            self._img = cv2.imread(self._img_path)
            self._img_og = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
            self._img_height, self._img_width, _ = image.shape
            return True

    def iris_localization(self):
        image = cv2.GaussianBlur(self._img, (15, 15), 0)
        edges = cv2.Canny(image, 50, 100)
        pupil_edge = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            2,
            minDist=70,
            param2=50,
            minRadius=20,
            maxRadius=60,
        )

        pupil_edge = np.uint16(np.around(pupil_edge))
        for value in pupil_edge[0, :]:
            self._pupil_center_x = value[0]
            self._pupil_center_y = value[1]
            self._pupil_radius = value[2]

        blank = np.zeros_like(self._img_og)
        mask_pupil = cv2.circle(
            blank,
            (self._pupil_center_x, self._pupil_center_y),
            self._pupil_radius,
            (255, 255, 255),
            -1,
        )
        mask_pupil = cv2.bitwise_not(mask_pupil)
        image = cv2.bitwise_and(self._img_og, mask_pupil)
        mask_background = cv2.circle(
            blank,
            (self._pupil_center_x, self._pupil_center_y),
            self._pupil_radius * 3,
            (255, 255, 255),
            -1,
        )
        image = cv2.bitwise_and(image, mask_background)

        image = cv2.equalizeHist(image)
        image = cv2.GaussianBlur(image, (15, 15), 0)

        # cv2.imshow('detected circles',test)
        # cv2.waitKey(0)

        edges = cv2.Canny(image, 50, 60)

        # cv2.imshow('detected circles',edges)
        # cv2.waitKey(0)

        iris_edge = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            2,
            minDist=1000,
            param2=10,
            minRadius=self._pupil_radius + 20,
            maxRadius=((self._pupil_radius * 3) - 10),
        )

        iris_edge = np.uint16(np.around(iris_edge))
        for value in iris_edge[0, :]:
            self._iris_center_x = value[0]
            self._iris_center_y = value[1]
            self._iris_radius = value[2]

        if self._iris_radius + self._iris_center_x > self._img_width:
            self._iris_radius = self._img_width - self._iris_center_x
        if self._iris_radius - self._iris_center_x < 0:
            self._iris_radius = self._iris_center_x
        if self._iris_radius + self._iris_center_y > self._img_height:
            self._iris_radius = self._img_height - self._iris_center_x
        if self._iris_radius - self._iris_center_y < 0:
            self._iris_radius = self._iris_center_y

        # cv2.circle(image,(iris[0],iris[1]),iris[2],color = (0,255, 0), thickness = 2)
        # cv2.circle(image,(pupil[0],pupil[1]),pupil[2],color = (0,255, 0), thickness = 2)

        # cv2.imshow('detected circles',image)
        # cv2.waitKey(0)

        blank = np.zeros_like(self._img_og)
        mask_pupil = cv2.circle(
            blank,
            (self._pupil_center_x, self._pupil_center_y),
            self._pupil_radius,
            (255, 255, 255),
            -1,
        )
        mask_pupil = cv2.bitwise_not(mask_pupil)
        image = cv2.bitwise_and(self._img_og, mask_pupil)
        mask_background = cv2.circle(
            blank,
            (self._iris_center_x, self._iris_center_y),
            self._iris_radius,
            (255, 255, 255),
            -1,
        )
        image = cv2.bitwise_and(image, mask_background)

        # cv2.imshow('detected circles',result)
        # cv2.waitKey(0)

        self._img = image

    def iris_normalization(self):
        # theta = np.arange(
        #     0.00, np.pi * 2, 0.01
        # )  # array of columns in final image (goes to 6.28 because period is 2 pi)
        # r = np.arange(
        #     0, self._iris_radius - self._pupil_radius, 1
        # )  # array of rows in final image
        # image = self._img
        # final = image[
        #     self._iris_center_x
        #     - self._iris_radius : self._iris_center_x
        #     + self._iris_radius,
        #     self._iris_center_y
        #     - self._iris_radius : self._iris_center_y
        #     + self._iris_radius,
        # ]
        # # cv2.imshow('detected circles',final)
        # # cv2.waitKey(0)
        # cartesian_img = np.empty(
        #     shape=[self._iris_radius - self._pupil_radius, self._iris_radius * 2, 3]
        # )  # empty array of dimensions of final image
        # m = interp1d(
        #     [np.pi * 2, 0], [0, self._iris_radius * 2]
        # )  # interpolate location on x axis
        # # interpolation between x = values from 0 and 6.28 and y = 0 to width of final image

        # # calculate all pixel values for normalized cartesian image
        # for z in r:
        #     i = z + self._pupil_radius
        #     for j in theta:
        #         # x = rcos(theta)
        #         polarX = int((i * np.cos(j)) + self._iris_radius)
        #         # y = rsin(theta)
        #         polarY = int((i * np.sin(j)) + self._iris_radius)
        #         cartesian_img[z][int(m(j) - 1)] = final[polarY][polarX]

        # # cartesian_img = cartesian_img[:][pupil[2]:iris[2]] #patch
        # cartesian_img = cartesian_img.astype("uint8")
        # self._img = np.asarray(cartesian_img)
        # # cv2.imshow('detected circles', img)
        # # cv2.waitKey(0)
        # # plt.imshow(self._img)
        # # plt.show()

        theta = np.arange(
            0.00, np.pi * 2, 0.01
        )  # array of columns in final image (goes to 6.28 because period is 2 pi)
        r = np.arange(0, int(self._img.shape[0] / 2), 1)

        # print(r.size)
        cartesian_img = np.empty(
            shape=[r.size, int(self._img.shape[1]), 3]
        )  # empty array of dimensions of final image
        # print(cartesian_img.shape)

        m = interp1d(
            [np.pi * 2, 0], [0, int(self._img.shape[1])]
        )  # interpolate location on x axis
        # interpolation between x = values from 0 and 6.28 and y = 0 to width of final image

        # calculate all pixel values for normalized cartesian image
        for i in r:
            # i += pupil[2]
            for j in theta:
                # x = rcos(theta)
                polarX = int((i * np.cos(j)) + self._img.shape[1] / 2)
                # y = rsin(theta)
                polarY = int((i * np.sin(j)) + self._img.shape[0] / 2)

                cartesian_img[i][int(m(j) - 1)] = self._img[polarY][polarX]

        cartesian_img = cartesian_img[:][
            self._pupil_radius : self._iris_radius
        ]  # patch
        cartesian_img = cartesian_img.astype("uint8")
        self._img = np.asarray(cartesian_img)

    def image_enhancement(self):
        image = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)
        # enhanced_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # plt.imshow(enhanced_img)

        # noise reduction
        ret, eyelid = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)
        # plt.imshow(eyelid)
        eyelid = cv2.GaussianBlur(eyelid, (15, 15), 0)
        eyelid = cv2.Canny(eyelid, 50, 100)
        # plt.imshow(eyelid)

        eyelid_circ = cv2.HoughCircles(
            eyelid,
            cv2.HOUGH_GRADIENT,
            2,
            minDist=500,
            param2=10,
            minRadius=0,
            maxRadius=100,
        )

        eyelid_circ = np.uint16(np.around(eyelid_circ))
        for value in eyelid_circ[0, :]:
            eyelid_circ_x = value[0]
            eyelid_circ_y = value[1]
            eyelid_circ_radius = value[2]

        # eyelid_circ

        base = np.zeros_like(image)
        mask_eyelid = cv2.circle(
            base,
            (eyelid_circ_x, eyelid_circ_y),
            eyelid_circ_radius,
            color=(0, 0, 0),
            thickness=2,
        )
        res2 = cv2.bitwise_not(mask_eyelid)
        # plt.imshow(res2)

        mask_eyelid = cv2.bitwise_or(image, mask_eyelid)
        result = cv2.bitwise_and(image, mask_eyelid)
        mask_background = cv2.circle(
            base,
            (eyelid_circ_x, eyelid_circ_y),
            eyelid_circ_radius,
            (255, 255, 255),
            -1,
        )
        result = cv2.bitwise_or(result, mask_background)
        # plt.imshow(result)

        ret, more_mask = cv2.threshold(result, 190, 255, cv2.THRESH_BINARY)
        # plt.imshow(more_mask)

        xd = cv2.bitwise_not(more_mask)
        xd = cv2.bitwise_and(image, xd)
        # plt.imshow(xd)
        self._img = xd
        # cv2.imshow('detected circles', self._img)
        # cv2.waitKey(0)

    def feature_extraction(self):
        filter_size = 9
        height = np.fix(filter_size / 2)
        width = np.fix(filter_size / 2)
        deltaX = 4.5
        deltaY = 1.5
        f = 1 / deltaY

        # constructing the kernel
        gabor_filter = np.zeros([filter_size, filter_size])
        for i in range(int(-height), int(height) + 1, 1):
            for j in range(int(-width), int(width) + 1, 1):
                # normalizing_factor = (1 / (2 * np.pi * deltaX * deltaY))
                g = np.exp(
                    (-0.5) * ((i**2 / deltaX**2) + (j**2 / deltaY**2))
                )  # Gabor function
                m1 = np.cos(
                    2 * np.pi * f * (np.sqrt((i**2) + (j**2)))
                )  # modulating function for two channels
                gabor_filter[i, j] = g * m1

        # plt.imshow(gabor_filter)

        # applying the kernel
        filtered_img = cv2.filter2D(self._img, cv2.CV_8UC3, gabor_filter)
        # plt.imshow(filtered_img)

        # standardizing image shape for classifier input
        standard_size_img = cv2.resize(filtered_img, (250, 60))
        # plt.imshow(standard_size_img)
        # standard_size_img.shape
        return standard_size_img


def iris_recognition(path):
    iris = iris_detection(path)
    iris.load_image()
    iris.iris_localization()
    iris.iris_normalization()
    iris.image_enhancement()
    values = iris.feature_extraction()

    return values


################
# Implementation
################

folder = "CASIA Iris Image Database (version 1.0)"
train_df = pd.DataFrame(columns=["id", "eye1", "eye2", "eye3"])
test_df = pd.DataFrame(columns=["id", "eye1", "eye2", "eye3", "eye4"])

for person in os.listdir(folder):
    train_info = {"id": person}
    test_info = {"id": person}
    for file in os.listdir(folder + "/" + person):
        if file == "1":
            for picture in os.listdir(folder + "/" + person + "/" + "1"):
                if picture[-4:] != ".bmp":
                    continue
                path = folder + "/" + person + "/" + "1" + "/" + picture
                train_info["eye" + picture[-5]] = path
        if file == "2":
            for picture in os.listdir(folder + "/" + person + "/" + "2"):
                if picture[-4:] != ".bmp":
                    continue
                path = folder + "/" + person + "/" + "2" + "/" + picture
                test_info["eye" + picture[-5]] = path

    train_df = train_df.append(train_info, ignore_index=True)
    test_df = test_df.append(test_info, ignore_index=True)


train_df = train_df.sort_values(by=["id"])
test_df = test_df.sort_values(by=["id"])

formatted_train_df = pd.DataFrame(columns=["id", "image"])
formatted_test_df = pd.DataFrame(columns=["id", "image"])

for index, row in train_df.iterrows():
    id = row["id"]
    print(id)
    eye1 = iris_recognition(row["eye1"])
    new_row = {"id": id, "image": eye1}
    formatted_train_df = formatted_train_df.append(new_row, ignore_index=True)

    eye2 = iris_recognition(row["eye2"])
    new_row = {"id": id, "image": eye2}
    formatted_train_df = formatted_train_df.append(new_row, ignore_index=True)

    eye3 = iris_recognition(row["eye3"])
    new_row = {"id": id, "image": eye3}
    formatted_train_df = formatted_train_df.append(new_row, ignore_index=True)

for index, row in test_df.iterrows():
    id = row["id"]

    eye4 = iris_recognition(row["eye4"])
    new_row = {"id": id, "image": eye4}
    formatted_test_df = formatted_test_df.append(new_row, ignore_index=True)

train_y = np.asarray(formatted_train_df["id"])
train_x = formatted_train_df.drop(columns=["id"])
train_x = train_x.reshape((len(train_x), 1))
train_y = train_y.reshape((len(train_y), 1))

test_y = np.asarray(formatted_test_df["id"])
test_x = np.asarray(formatted_test_df["id"])
test_x = test_x.reshape((len(train_x), 1))
test_y = test_y.reshape((len(train_y), 1))

LDA = LinearDiscriminantAnalysis()
train_x = LDA.fit_transform(train_x, train_y)
test_x = LDA.transform(test_x)

model = NearestCentroid()
model.fit(train_x, train_y)
predictions = model.predit(test_x)

accuracy = accuracy_score(train_y, predictions)
