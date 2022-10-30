import cv2
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

image = cv2.imread('test_4.bmp')


test = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



test = cv2.GaussianBlur(test,(15,15), 0)
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

# test = cv2.equalizeHist(result)
# test = cv2.GaussianBlur(test,(15,15),0)

cv2.imshow('detected circles',result)
cv2.waitKey(0)

edges = cv2.Canny(test, 50, 60)

cv2.imshow('detected circles',edges)
cv2.waitKey(0)

iris = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,2,minDist = 1000,
                            param2=10,minRadius=pupil[2]+20,maxRadius=((pupil[2]*3)-10))

iris = np.uint16(np.around(iris))
for value in iris[0,:]:
    iris_x = value[0]
    iris_y = value[1]
    radius = value[2]
iris = np.array([center_x, center_y, radius])

# cv2.circle(image,(iris[0],iris[1]),iris[2],color = (0,255, 0), thickness = 2)
# cv2.circle(image,(pupil[0],pupil[1]),pupil[2],color = (0,255, 0), thickness = 2)

# cv2.imshow('detected circles',image)
# cv2.waitKey(0)

base = np.zeros_like(image)
mask_pupil = cv2.circle(base, (pupil[0], pupil[1]), pupil[2], (255,255,255), -1)
mask_pupil = cv2.bitwise_not(mask_pupil)
result = cv2.bitwise_and(image, mask_pupil)
mask_iris = cv2.circle(base, (iris[0], iris[1]), iris[2], (255,255,255), -1)
result = cv2.bitwise_and(result, mask_iris)

cv2.imshow('detected circles',result)
cv2.waitKey(0)

###########################
iris_width = iris[2] - pupil[2]
iris_diameter = iris[2]*2

theta = np.arange(0.00, np.pi * 2, 0.01) #array of columns in final image (goes to 6.28 because period is 2 pi)
r = np.arange(0, iris[2]- pupil[2], 1) #array of rows in final image
final = result[iris[1] - iris[2] : iris[1] + iris[2], iris[0] - iris[2] : iris[0] + iris[2]]
cv2.imshow('detected circles',final)
cv2.waitKey(0)
print(final.shape)
print(iris[2]*2)
cartesian_img = np.empty(shape = [iris_width, iris_diameter, 3]) # empty array of dimensions of final image
print(cartesian_img.shape)
# plt.imshow(cartesian_img)
# plt.show()
m = interp1d([np.pi * 2, 0],[0, iris_diameter]) #interpolate location on x axis
    # interpolation between x = values from 0 and 6.28 and y = 0 to width of final image

# calculate all pixel values for normalized cartesian image
for z in r:
    i = z + pupil[2]
    for j in theta:
        # x = rcos(theta)
        polarX = int((i * np.cos(j)) + iris_diameter / 2)
        # y = rsin(theta)
        polarY = int((i * np.sin(j)) + iris_diameter / 2)
        cartesian_img[z][int(m(j) - 1)] = final[polarY][polarX]


# cartesian_img = cartesian_img[:][pupil[2]:iris[2]] #patch
cartesian_img = cartesian_img.astype('uint8')
img = np.asarray(cartesian_img)
# cv2.imshow('detected circles', img)
# # cv2.waitKey(0)
plt.imshow(cartesian_img)
plt.show()
########################
test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
test = cv2.equalizeHist(test)
# enhanced = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
# plt.imshow(test)
# plt.show()
cv2.imshow('detected circles', test)
cv2.waitKey(0)


