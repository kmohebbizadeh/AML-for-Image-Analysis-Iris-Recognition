import cv2
import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

class iris_detection():
    def __init__(self, image_path):
        self._img = None
        self._img_og = None
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
            self._img_og = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
            return True
    def iris_localization(self):
        image = cv2.GaussianBlur(self._img,(15,15), 0)
        edges = cv2.Canny(image,50,100)
        pupil_edge = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,2, minDist = 70,
                                    param2=50,minRadius=20,maxRadius=60)

        pupil_edge = np.uint16(np.around(pupil_edge))
        for value in pupil_edge[0,:]:
            self._pupil_center_x = value[0]
            self._pupil_center_y = value[1]
            self._pupil_radius = value[2]

        blank = np.zeros_like(self._img_og)
        mask_pupil = cv2.circle(blank, (self._pupil_center_x, self._pupil_center_y), self._pupil_radius, (255,255,255), -1)
        mask_pupil = cv2.bitwise_not(mask_pupil)
        image = cv2.bitwise_and(self._img_og, mask_pupil)
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
                                    param2=10,minRadius=self._pupil_radius+20,maxRadius=((self._pupil_radius*3)-10))

        iris_edge = np.uint16(np.around(iris_edge))
        for value in iris_edge[0,:]:
            self._iris_center_x = value[0]
            self._iris_center_y = value[1]
            self._iris_radius = value[2]

        # cv2.circle(image,(iris[0],iris[1]),iris[2],color = (0,255, 0), thickness = 2)
        # cv2.circle(image,(pupil[0],pupil[1]),pupil[2],color = (0,255, 0), thickness = 2)

        # cv2.imshow('detected circles',image)
        # cv2.waitKey(0)

        blank = np.zeros_like(self._img_og)
        mask_pupil = cv2.circle(blank, (self._pupil_center_x, self._pupil_center_y), self._pupil_radius, (255,255,255), -1)
        mask_pupil = cv2.bitwise_not(mask_pupil)
        image = cv2.bitwise_and(self._img_og, mask_pupil)
        mask_background = cv2.circle(blank, (self._iris_center_x, self._iris_center_y), self._iris_radius, (255,255,255), -1)
        image = cv2.bitwise_and(image, mask_background)

        # cv2.imshow('detected circles',result)
        # cv2.waitKey(0)

        self._img = image

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
        self._img = np.asarray(cartesian_img)
        # cv2.imshow('detected circles', img)
        # cv2.waitKey(0)
        # plt.imshow(img)
        # plt.show()

    def image_enhancement(self):
        image = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)
        # enhanced_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # plt.imshow(enhanced_img)
        self._img = image

    def feature_extraction(self):
        pass

    def dimensionality_reduction(self):
        #return values
        pass



def iris_recognition(path):
    iris = iris_detection(path)
    iris.load_image()
    iris.iris_localization()
    iris.iris_normalization()
    iris.image_enhancement()
    iris.feature_extraction()
    values = iris.dimensionality_reduction()

    return values




################
#Implementation
################

folder = 'CASIA Iris Image Database (version 1.0)'
train_df = pd.DataFrame(columns=['id', 'eye1', 'eye2', 'eye3'])
test_df = pd.DataFrame(columns=['id', 'eye1', 'eye2', 'eye3', 'eye4'])

for person in os.listdir(folder):
    train_info = {'id': person}
    test_info = {'id': person}
    for file in os.listdir(folder + '/' + person):
        if file == '1':
            for picture in os.listdir(folder + '/' + person + '/' + '1'):
                if picture[-4:] != '.bmp':
                    continue
                path = folder + '/' + person + '/' + '1' + '/' + picture
                train_info['eye'+picture[-5]] = path
        if file == '2':
            for picture in os.listdir(folder + '/' + person + '/' + '2'):
                if picture[-4:] != '.bmp':
                    continue
                path = folder + '/' + person + '/' + '2' + '/' + picture
                test_info['eye'+picture[-5]] = path

    train_df = train_df.append(train_info, ignore_index=True)
    test_df = test_df.append(test_info, ignore_index=True)


train_df = train_df.sort_values(by=['id'])
test_df = test_df.sort_values(by=['id'])

formatted_train_df = pd.DataFrame(columns=['id', 'image'])
formatted_test_df = pd.DataFrame(columns=['id', 'image'])

for index, row in train_df.iterrows():
    id = row['id']

    eye1 = iris_recognition(row['eye1'])
    new_row = {'id': id, 'image': eye1}
    formatted_train_df = train_df.append(new_row, ignore_index=True)

    eye2 = iris_recognition(row['eye2'])
    new_row = {'id': id, 'image': eye2}
    formatted_train_df = train_df.append(new_row, ignore_index=True)

    eye3 = iris_recognition(row['eye3'])
    new_row = {'id': id, 'image': eye3}
    formatted_train_df = train_df.append(new_row, ignore_index=True)

for index, row in test_df.iterrows():
    id = row['id']

    eye4 = iris_recognition(row['eye4'])
    new_row = {'id': id, 'image': eye4}
    formatted_test_df = test_df.append(new_row, ignore_index=True)

train_y = np.asarray(formatted_train_df['id'])
train_x = formatted_train_df.drop(columns=['id'])
train_x = train_x.reshape((len(train_x), 1))
train_y = train_y.reshape((len(train_y), 1))

test_y = np.asarray(formatted_test_df['id'])
test_x = np.asarray(formatted_test_df['id'])
test_x = test_x.reshape((len(train_x), 1))
test_y = test_y.reshape((len(train_y), 1))

model = NearestCentroid()
model.fit(train_x, train_y)
predictions = model.predit(test_x)

accuracy = accuracy_score(train_y, predictions)














