import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('/Users/patrickutz/cs-projects/python/traffic-cone-OpenCV-detection/images/1.jpg')

cv2.imshow('ImageWindow', img)
cv2.waitKey(0)