################
# Load Libraries
################
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestCentroid
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.ndimage import generic_filter
import warnings

warnings.filterwarnings("ignore")

################
# Define Functions
################

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

        if type(image) is type(None): # test to make sure image read correctly
            return False
        else: # save images and specs for use in functions
            self._img = cv2.imread(self._img_path)
            self._img_og = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
            self._img_height, self._img_width, _ = image.shape
            return True

    def iris_localization(self):
        # prep the image for detection
        image = cv2.GaussianBlur(self._img, (15, 15), 0) # 15x15 is a standard kernel

        # detect the pupil
        edges = cv2.Canny(image, 50, 100) # 50, 100 remove some of the noise, but retain the sharp lines in the image
        pupil_edge = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            2, # standard
            minDist=1000, # large so only one circle is selected
            param2=50, # large so function only picks the strongest circular shape
            minRadius=20, # avoids picking up the smaller circles in the image
            maxRadius=70, # avoids picking large circles in the image all pupils are in this range
        )

        pupil_edge = np.uint16(np.around(pupil_edge))

        # mask the image for easier iris detection
        for value in pupil_edge[0, :]: # save pupil values
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
        image = cv2.bitwise_and(self._img_og, mask_pupil) # mask off the pupil
        mask_background = cv2.circle(
            blank,
            (self._pupil_center_x, self._pupil_center_y),
            self._pupil_radius * 3,
            (255, 255, 255),
            -1,
        )
        image = cv2.bitwise_and(image, mask_background) # mask 3x the pupils radius to narrow the iris search

        # prepare image for iris detection
        image = cv2.equalizeHist(image) # light colored iris's need more contrast to detect
        image = cv2.GaussianBlur(image, (25, 25), 0) # stronger blur to remove noise

        # iris edge detection
        edges = cv2.Canny(image, 0, 10) # because we have a strong blur, we want to pull the detailed edges of the image

        iris_edge = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            2,
            minDist=1000, # only want one circle
            param2=50, # very strong circle must be detected
            minRadius=self._pupil_radius + 20, # larger than the pupil mask
            maxRadius=((self._pupil_radius * 3) - 10), # smaller than the outer mask
        )

        iris_edge = np.uint16(np.around(iris_edge))

        # set iris around pupil center for normalization purposes (equal width ring)
        for value in iris_edge[0, :]:
            self._iris_center_x = self._pupil_center_x
            self._iris_center_y = self._pupil_center_y
            self._iris_radius = value[2]

        alt_radius = min(
            [
                (self._img_width - self._iris_center_x),
                self._iris_center_x,
                (self._img_height - self._iris_center_y),
                self._iris_center_y,
            ]
        )

        # check that the iris detected does not go outside the edge of the image
        if alt_radius < self._iris_radius:
            self._iris_radius = alt_radius

        # mask the image around the pupil and iris
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
        # center the image around the iris
        image = self._img
        final = image[
            self._iris_center_y
            - self._iris_radius : self._iris_center_y
            + self._iris_radius,
            self._iris_center_x
            - self._iris_radius : self._iris_center_x
            + self._iris_radius,
        ]

        # transform image to rectangle
        theta = np.arange(0.00, np.pi * 2, 0.01)
        r = np.arange(0, self._iris_radius, 1)

        cartesian_img = np.empty(
            shape=[self._iris_radius * 2, self._iris_radius * 2, 3]
        )

        m = interp1d([np.pi * 2, 0], [0, self._iris_radius * 2])

        for z in r:
            for j in theta:
                polarX = int((z * np.cos(j)) + self._iris_radius)
                polarY = int((z * np.sin(j)) + self._iris_radius)
                cartesian_img[z][int(m(j) - 1)] = final[polarY][polarX]
        cartesian_img = cartesian_img[:][self._pupil_radius : self._iris_radius]
        cartesian_img = cartesian_img.astype("uint8")
        self._img = np.asarray(cartesian_img)

    def image_enhancement(self):
        # prepare the image for feature extraction
        image = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        eyelid = cv2.GaussianBlur(image, (15, 15), 0)

        # run edge detection to find eyelids
        eyelid = cv2.Canny(eyelid, 40, 50)

        eyelid_circ = cv2.HoughCircles(
            eyelid,
            cv2.HOUGH_GRADIENT,
            2,
            minDist=120, # we want the possibility of finding two eyelids
            param2=40, # dont want noise circles so this value is set high
            minRadius=15, # avoid small circles detected
            maxRadius=50, # avoid large portions of the image removed
        )

        # create mask from detected circles to remove eyelids
        base = np.zeros_like(image)

        if eyelid_circ is not None:
            eyelid_circ = np.uint16(np.around(eyelid_circ))
            for value in eyelid_circ[0, :]:
                eyelid_circ_x = value[0]
                eyelid_circ_y = value[1]
                eyelid_circ_radius = value[2]
                new_image = cv2.circle(
                    image,
                    (eyelid_circ_x, eyelid_circ_y),
                    eyelid_circ_radius,
                    color=(0, 0, 0),
                    thickness=2,
                )
                mask_background = cv2.circle(
                    base,
                    (eyelid_circ_x, eyelid_circ_y),
                    eyelid_circ_radius,
                    (255, 255, 255),
                    -1,
                )

            mask_eyelid = cv2.circle(
                new_image,
                (eyelid_circ_x, eyelid_circ_y),
                eyelid_circ_radius,
                color=(0, 0, 0),
                thickness=2,
            )

            # overlay mask with original image in a series of bitwise operators
            result = cv2.bitwise_and(new_image, mask_eyelid)
            result = cv2.bitwise_or(result, mask_background)
            ret, white_mask = cv2.threshold(result, 254, 255, cv2.THRESH_BINARY)
            final = cv2.bitwise_not(white_mask)
            final = cv2.bitwise_and(new_image, final)

            # histogram equalization for iris detail enhancement
            final = cv2.equalizeHist(final)
            self._img = final

            # standardize size of output
            self._img = cv2.resize(self._img, (512, 48))

        else: # if no circles, just equalize and resize the image
            self._img = cv2.equalizeHist(image)
            self._img = cv2.resize(self._img, (512, 48))

    def feature_extraction(self):
        deltaX1 = 3
        deltaX2 = 4.5
        deltaY = 1.5

        # kernel for first channel
        kernel1 = cv2.getGaborKernel(
            ksize=(9, 9),
            sigma=deltaX1,
            theta=0,
            lambd=deltaY,
            gamma=1,
            psi=0,
            ktype=cv2.CV_32F,
        )
        # kernel for second channel
        kernel2 = cv2.getGaborKernel(
            ksize=(9, 9),
            sigma=deltaX2,
            theta=0,
            lambd=deltaY,
            gamma=1,
            psi=0,
            ktype=cv2.CV_32F,
        )

        # applying kernels
        filtered_img1 = cv2.filter2D(self._img, cv2.CV_8UC3, kernel1)
        filtered_img2 = cv2.filter2D(self._img, cv2.CV_8UC3, kernel2)

        # apply mean and standard deviation filter to entire image
        vec = []
        block_size = 8
        kernel = np.array([[1,1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1]])

        means1 = generic_filter(filtered_img1, np.mean, footprint = kernel)
        sds1 = generic_filter(filtered_img1, np.std, footprint = kernel)

        means2 = generic_filter(filtered_img2, np.mean, footprint = kernel)
        sds2 = generic_filter(filtered_img2, np.std, footprint = kernel)

        # select values at the center of the kernel
        for i in range(0, filtered_img1.shape[0], block_size):
            for j in range(0, filtered_img1.shape[1], block_size):
                # when selecting value we want to select the middle of the block hence i+4, j+4
                vec.append(means1[i+4,j+4])
                vec.append(sds1[i+4,j+4])

        for i in range(0, filtered_img2.shape[0], block_size):
            for j in range(0, filtered_img2.shape[1], block_size):
                vec.append(means2[i+4,j+4])
                vec.append(sds2[i+4,j+4])
        
        vec = np.asarray(vec)

        return vec


def iris_recognition(path):
    # image processing order
    iris = iris_detection(path)
    iris.load_image()
    iris.iris_localization()
    iris.iris_normalization()
    iris.image_enhancement()
    feature_vector = iris.feature_extraction()

    return feature_vector

################
# Implementation on Data
################

# initialize data frame for training and test set image paths
folder = "CASIA Iris Image Database (version 1.0)"
train_df = pd.DataFrame(columns=["id", "eye1", "eye2", "eye3"])
test_df = pd.DataFrame(columns=["id", "eye1", "eye2", "eye3", "eye4"])

# iterate through directory and save paths to the dataframe
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

# sort for easier comparison and testing
train_df = train_df.sort_values(by=["id"])
test_df = test_df.sort_values(by=["id"])

# initialize values dataframe
formatted_train_df = pd.DataFrame(columns=["id"])
formatted_test_df = pd.DataFrame(columns=["id"])

for i in range(1536):
    formatted_train_df[i] = None
    formatted_test_df[i] = None

# process each image and store in train dataframe
entry = 0
for index, row in train_df.iterrows():
    id = row["id"]

    eye1 = iris_recognition(row["eye1"])
    new_row = {"id": id}
    formatted_train_df = formatted_train_df.append(new_row, ignore_index=True)
    for i in range(len(eye1)):
        formatted_train_df.loc[entry].at[i] = eye1[i]
    entry += 1

    eye2 = iris_recognition(row["eye2"])
    new_row = {"id": id}
    formatted_train_df = formatted_train_df.append(new_row, ignore_index=True)
    for i in range(len(eye2)):
        formatted_train_df.loc[entry].at[i] = eye2[i]
    entry += 1

    eye3 = iris_recognition(row["eye3"])
    new_row = {"id": id}
    formatted_train_df = formatted_train_df.append(new_row, ignore_index=True)
    for i in range(len(eye3)):
        formatted_train_df.loc[entry].at[i] = eye3[i]
    entry += 1
    if entry == 12:
        break

# process each image and store in test dataframe, only need fourth image for testing
entry = 0
for index, row in test_df.iterrows():
    id = row["id"]
    eye4 = iris_recognition(row["eye4"])
    new_row = {"id": id}
    formatted_test_df = formatted_test_df.append(new_row, ignore_index=True)
    for i in range(len(eye4)):
        formatted_test_df.loc[entry].at[i] = eye4[i]
    entry += 1
    if entry == 2:
        break

# standardize dataframes for sklearn
train_y = np.asarray(formatted_train_df["id"])
train_x = formatted_train_df.drop(columns=["id"])
train_x = np.asarray(train_x)

test_y = np.asarray(formatted_test_df["id"])
test_x = formatted_test_df.drop(columns=["id"])
test_x = np.asarray(test_x)

# dimensionality reduction for both train and test x sets
LDA = LinearDiscriminantAnalysis()
train_x = LDA.fit_transform(train_x, train_y)
test_x = LDA.transform(test_x)

# fit the model using the processed train data
model = NearestCentroid()
model.fit(train_x, train_y)

# predict based on the test data
predictions = model.predict(test_x)

# calculate metrics for model evaluation
accuracy = metrics.accuracy_score(test_y, predictions)
roc_ovr = metrics.roc_auc_score(test_y, predictions, multi_class="ovr", average="macro")
roc_ovo = metrics.roc_auc_score(test_y, predictions, multi_class="ovo", average="macro")

print("ROC score (OVR): ", roc_ovr)
print("ROC score (OVO): ", roc_ovr)
print("Accuracy (CRR): ", accuracy)
