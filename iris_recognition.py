import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.ndimage import generic_filter
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
            minDist=100,
            param2=50,
            minRadius=20,
            maxRadius=70,
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
        image = cv2.GaussianBlur(image, (25, 25), 0)

        edges = cv2.Canny(image, 0, 10)

        iris_edge = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            2,
            minDist=1000,
            param2=50,
            minRadius=self._pupil_radius + 20,
            maxRadius=((self._pupil_radius * 3) - 10),
        )

        iris_edge = np.uint16(np.around(iris_edge))
        for value in iris_edge[0, :]:
            self._iris_center_x = self._pupil_center_x
            self._iris_center_y = self._pupil_center_y
            self._iris_radius = value[2]

        alt_radius = min([(self._img_width - self._iris_center_x), self._iris_center_x, (self._img_height - self._iris_center_y), self._iris_center_y ])

        if alt_radius < self._iris_radius:
            self._iris_radius = alt_radius

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

        self._img = image

    def iris_normalization(self):
        image = self._img
        final = image[
            self._iris_center_y - self._iris_radius : self._iris_center_y + self._iris_radius,
            self._iris_center_x - self._iris_radius : self._iris_center_x + self._iris_radius
        ]
        theta = np.arange(0.00, np.pi * 2, 0.01)
        r = np.arange(0, self._iris_radius, 1)

        cartesian_img = np.empty(shape=[self._iris_radius*2, self._iris_radius*2, 3])

        m = interp1d([np.pi * 2, 0], [0, self._iris_radius*2])

        for z in r:
            for j in theta:
                polarX = int((z * np.cos(j)) + self._iris_radius)
                polarY = int((z * np.sin(j)) + self._iris_radius)
                cartesian_img[z][int(m(j) - 1)] = final[polarY][polarX]
        cartesian_img = cartesian_img[:][self._pupil_radius: self._iris_radius]
        cartesian_img = cartesian_img.astype("uint8")
        self._img = np.asarray(cartesian_img)

        # cv2.imshow('detected circles', self._img)
        # cv2.waitKey(0)

    def image_enhancement(self):
        image = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        # image = cv2.equalizeHist(image)
        # cv2.imshow('detected circles', image)
        # cv2.waitKey(0)
        # self._img = image

        # enhanced_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # plt.imshow(enhanced_img)

        # # noise reduction
        ret, eyelid = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
        # cv2.imshow('detected circles', eyelid)
        # cv2.waitKey(0)
        # # plt.imshow(eyelid)
        eyelid = cv2.GaussianBlur(image, (15, 15), 0)
        eyelid = cv2.Canny(eyelid, 40, 50)
        cv2.imshow('detected circles', eyelid)
        cv2.waitKey(0)
        # # plt.imshow(eyelid)
        #
        eyelid_circ = cv2.HoughCircles(
            eyelid,
            cv2.HOUGH_GRADIENT,
            2,
            minDist=120,
            param2=40,
            minRadius=15,
            maxRadius=50,)

        eyelid_circ = np.uint16(np.around(eyelid_circ))
        for value in eyelid_circ[0, :]:
            eyelid_circ_x = value[0]
            eyelid_circ_y = value[1]
            eyelid_circ_radius = value[2]
            image = cv2.circle(
            image,
            (eyelid_circ_x, eyelid_circ_y),
            eyelid_circ_radius,
            color=(0, 0, 0),
            thickness=2)

        # eyelid_circ

        # base = np.zeros_like(image)
        # mask_eyelid = cv2.circle(
        #     image,
        #     (eyelid_circ_x, eyelid_circ_y),
        #     eyelid_circ_radius,
        #     color=(0, 0, 0),
        #     thickness=2,
        # )
        # cv2.imshow('detected circles', image)
        # cv2.waitKey(0)
        # res2 = cv2.bitwise_not(mask_eyelid)
        # # plt.imshow(res2)
        #
        # mask_eyelid = cv2.bitwise_or(image, mask_eyelid)
        # result = cv2.bitwise_and(image, mask_eyelid)
        # mask_background = cv2.circle(
        #     base,
        #     (eyelid_circ_x, eyelid_circ_y),
        #     eyelid_circ_radius,
        #     (255, 255, 255),
        #     -1,
        # )
        # result = cv2.bitwise_or(result, mask_background)
        # # plt.imshow(result)
        #
        # ret, more_mask = cv2.threshold(result, 190, 255, cv2.THRESH_BINARY)
        # # plt.imshow(more_mask)
        #
        # xd = cv2.bitwise_not(more_mask)
        # xd = cv2.bitwise_and(image, xd)
        # plt.imshow(xd)
        # self._img = xd
        # cv2.imshow('detected circles', self._img)
        # cv2.waitKey(0)

        # standardize size here
    def feature_extraction(self):
        # filter_size = 9
        # height = np.fix(filter_size / 2)
        # width = np.fix(filter_size / 2)
        # deltaX1 = 3
        # deltaX2 = 4.5
        # deltaY = 1.5
        # f1 = 1 / deltaX1
        # f2 = 1 / deltaX2
        #
        #
        # # constructing the kernel
        # gabor_filter = np.zeros([filter_size, filter_size])
        # gabor_filter2 = np.zeros([filter_size, filter_size])
        #
        # for i in range(int(-height), int(height) + 1, 1):
        #     for j in range(int(-width), int(width) + 1, 1):
        #         # normalizing_factor = (1 / (2 * np.pi * deltaX * deltaY))
        #         g1 = np.exp(
        #             (-0.5) * ((i**2 / deltaX1**2) + (j**2 / deltaY**2))
        #         )  # Gabor function
        #         g2 = np.exp(
        #             (-0.5) * ((i**2 / deltaX2**2) + (j**2 / deltaY**2))
        #         )  # Gabor function
        #         m1 = np.cos(
        #             2 * np.pi * f1 * (np.sqrt((i**2) + (j**2)))
        #         )  # modulating function for two channels
        #         m2 = np.cos(
        #             2 * np.pi * f2 * (np.sqrt((i**2) + (j**2)))
        #         )  # modulating function for two channels
        #         gabor_filter[i, j] = g1 * m1
        #         gabor_filter2[i, j] = g2 * m2
        #
        # # plt.imshow(gabor_filter)
        #
        # # applying the kernel
        # filtered_img1 = cv2.filter2D(self._img, cv2.CV_8UC3, gabor_filter)
        # filtered_img2 = cv2.filter2D(self._img, cv2.CV_8UC3, gabor_filter2)

        deltaX1 = 3
        deltaX2 = 4.5
        deltaY = 1.5
        f1 = 1 / deltaY

        kernel1 = cv2.getGaborKernel(ksize=(9,9),
                           sigma=deltaX1,
                           theta=0,
                           lambd=deltaY,
                           gamma=1,
                           psi=0,
                           ktype=cv2.CV_32F)
        kernel2 = cv2.getGaborKernel(ksize=(9,9),
                   sigma=deltaX2,
                   theta=0,
                   lambd=deltaY,
                   gamma=1,
                   psi=0,
                   ktype=cv2.CV_32F)

        filtered_img1 = cv2.filter2D(self._img, cv2.CV_8UC3, kernel1)
        filtered_img2 = cv2.filter2D(self._img, cv2.CV_8UC3, kernel2)
        cv2.imshow('detected circles', filtered_img1)
        cv2.waitKey(0)
        cv2.imshow('detected circles', filtered_img2)
        cv2.waitKey(0)

        # self._img = cv2.resize(self._img, (512, 48))
        # plt.imshow(standard_size_img)
        # plt.show()
        # print(standard_size_img.shape)
        # standard_size_img.shape
        # gray_img = cv2.cvtColor(standard_size_img, cv2.COLOR_BGR2GRAY)
        # gray_img = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        # vec = []
        # block_size = 8
        # kernel = [[1,1,1,1,1,1,1,1],
        #           [1,1,1,1,1,1,1,1],
        #           [1,1,1,1,1,1,1,1],
        #           [1,1,1,1,1,1,1,1],
        #           [1,1,1,1,1,1,1,1],
        #           [1,1,1,1,1,1,1,1],
        #           [1,1,1,1,1,1,1,1],
        #           [1,1,1,1,1,1,1,1]]
        # means = generic_filter(self._img, np.mean, footprint = kernel)
        # sds = generic_filter(self._img, np.std, footprint = kernel)
        # # print(sds.shape)
        # # print(means)
        # plt.imshow(means)
        # # print(type(means))
        # # print(len(sds))
        #
        # for i in range(0, self._img.shape[0], block_size):
        #     for j in range(0, self._img.shape[1], block_size):
        #         vec.append(means[i,j])
        #         vec.append(sds[i,j])
        #
        # vec = np.asarray(vec)
        # return vec


def iris_recognition(path):
    iris = iris_detection(path)
    iris.load_image()
    iris.iris_localization()
    iris.iris_normalization()
    iris.image_enhancement()
    # iris.feature_extraction()

    # return value
#
# iris_recognition("test_1.bmp")
# iris_recognition("test_2.bmp")
# iris_recognition("test_3.bmp")
# iris_recognition("test_4.bmp")
################
# Implementation
################

folder = "CASIA Iris Image Database (version 1.0)"
train_df = pd.DataFrame(columns=["id", "eye1", "eye2", "eye3"])
test_df = pd.DataFrame(columns=["id", "eye1", "eye2", "eye3", "eye4"])
#
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
#
formatted_train_df = pd.DataFrame(columns=["id"])
formatted_test_df = pd.DataFrame(columns=["id"])
#
for i in range(768):
    formatted_train_df[i] = None
    formatted_test_df[i] = None
#
entry = 0
# for index, row in train_df.iterrows():
#     id = row["id"]
#
#     eye1 = iris_recognition(row["eye1"])
#     new_row = {"id": id}
#     formatted_train_df = formatted_train_df.append(new_row, ignore_index=True)
#     for i in range(len(eye1)):
#         formatted_train_df.loc[entry].at[i] = eye1[i]
#     entry += 1
#
#     eye2 = iris_recognition(row["eye2"])
#     new_row = {"id": id}
#     formatted_train_df = formatted_train_df.append(new_row, ignore_index=True)
#     for i in range(len(eye2)):
#         formatted_train_df.loc[entry].at[i] = eye2[i]
#     entry += 1
#
#     eye3 = iris_recognition(row["eye3"])
#     new_row = {"id": id}
#     formatted_train_df = formatted_train_df.append(new_row, ignore_index=True)
#     for i in range(len(eye3)):
#         formatted_train_df.loc[entry].at[i] = eye3[i]
#     entry += 1
#     if entry == 3:
#         break
#
#
entry = 0
for index, row in test_df.iterrows():
    id = row["id"]
    eye4 = iris_recognition(row["eye4"])
    new_row = {"id": id}
    formatted_test_df = formatted_test_df.append(new_row, ignore_index=True)
    # for i in range(len(eye4)):
    #     formatted_test_df.loc[entry].at[i] = eye4[i]
    entry += 1
    if entry == 25:
        break

# train_y = np.asarray(formatted_train_df["id"])
# train_x = formatted_train_df.drop(columns=["id"])
# train_x = np.asarray(train_x)
#
# test_y = np.asarray(formatted_test_df["id"])
# test_x = formatted_test_df.drop(columns=["id"])
# test_x = np.asarray(test_x)
#
# LDA = LinearDiscriminantAnalysis()
# train_x = LDA.fit_transform(train_x, train_y)
# test_x = LDA.transform(test_x)
#
# model = NearestCentroid()
# model.fit(train_x, train_y)
# predictions = model.predict(test_x)
#
# accuracy = accuracy_score(test_y, predictions)
#
# print(accuracy)
