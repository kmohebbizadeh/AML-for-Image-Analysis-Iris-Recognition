import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir

image = cv2.imread('test_1.bmp')


test = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
test_og = cv2.equalizeHist(test)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# # LOCALIZATION
test = cv2.GaussianBlur(test,(15,15), 0)
#
edges = cv2.Canny(test,50,100)
pupil = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,2, minDist = 70,
                            param2=50,minRadius=20,maxRadius=60)

pupil = np.uint16(np.around(pupil))
for value in pupil[0,:]:
    center_x = value[0]
    center_y = value[1]
    radius = value[2]

pupil = np.array([center_x, center_y, radius])
base = np.zeros_like(test)
mask_pupil = cv2.circle(base, (pupil[0], pupil[1]), pupil[2], (255,255,255), -1)
mask_pupil = cv2.bitwise_not(mask_pupil)
result = cv2.bitwise_and(test, mask_pupil)

mask_background = cv2.circle(base, (pupil[0], pupil[1]), pupil[2]*3, (255,255,255), -1)
result = cv2.bitwise_and(result, mask_background)

# cv2.imshow('detected circles',result)
# cv2.waitKey(0)
test = cv2.equalizeHist(result)
test = cv2.GaussianBlur(test,(25,25),0)
# cv2.imshow('detected circles',test)
# cv2.waitKey(0)
edges = cv2.Canny(test, 40, 70)
# cv2.imshow('detected circles',edges)
# cv2.waitKey(0)
iris = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,2,minDist = 1000,
                            param2=10,minRadius=pupil[2]+20,maxRadius=((pupil[2]*3)-10))
print(iris)
iris = np.uint16(np.around(iris))
for value in iris[0,:]:
    iris_x = value[0]
    iris_y = value[1]
    radius = value[2]
iris = np.array([center_x, center_y, radius])
cv2.circle(image,(iris[0],iris[1]),iris[2],color = (0,255, 0), thickness = 2)
cv2.circle(image,(pupil[0],pupil[1]),pupil[2],color = (0,255, 0), thickness = 2)
cv2.imshow('detected circles',image)
cv2.waitKey(0)



