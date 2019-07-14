import numpy as np
import cv2

# load an color image
original_img = cv2.imread('/Users/patrickutz/cs-projects/python/traffic-cone-OpenCV-detection/images/1.jpg')
cv2.imshow('ImageWindow', original_img)
cv2.waitKey(0)

# convert the image to HSV because easier to represent color in
# HSV as opposed to in BGR 
hsv_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV Conversion', hsv_img)
cv2.waitKey(0)

# define range of orange traffic cone color in HSV
lower_orange = np.array([3, 100, 100])
upper_orange = np.array([24, 255, 255])

# Threshold the HSV image to get only bright orange colors
mask = cv2.inRange(hsv_img, lower_orange, upper_orange)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(original_img, original_img, mask= mask)

cv2.imshow('original_img',original_img)
cv2.waitKey(0)
cv2.imshow('mask',mask)
cv2.waitKey(0)
cv2.imshow('res',res)
cv2.waitKey(0)