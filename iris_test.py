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
test = cv2.medianBlur(test_og,7)

edges = cv2.Canny(test,50,70)

pupil = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=40,minRadius=10,maxRadius=70)

edges = cv2.Canny(test, 70, 150)
iris = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,40,
                            param2=35,minRadius=20,maxRadius=120)
circles = np.append(np.array(pupil), np.array(iris), axis=0)

for i in circles[0,:2,]:
    cv2.circle(image,(i[0],i[1]),i[2],color = (0,255, 0), thickness = 2)

cv2.imshow('detected circles',image)
cv2.waitKey(0)


