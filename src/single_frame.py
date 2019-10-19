import numpy as np
import cv2
import random as rng

rng.seed(12345)
# load a color image
original_img = cv2.imread('/Users/patrickutz/cs-projects/python/traffic-cone-OpenCV-detection/images/10.jpg')
# color_img = cv2.imread('/Users/patrickutz/cs-projects/python/traffic-cone-OpenCV-detection/images/1.jpg')
cv2.imshow('ImageWindow', original_img)
cv2.waitKey(0)

# convert the image to HSV because easier to represent color in
# HSV as opposed to in BGR 
hsv_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)

# define range of orange traffic cone color in HSV
lower_orange1 = np.array([0, 135, 135])
lower_orange2 = np.array([15, 255, 255])
upper_orange1 = np.array([159, 135, 80])
upper_orange2 = np.array([179, 255, 255])

# threshold the HSV image to get only bright orange colors
imgThreshLow = cv2.inRange(hsv_img, lower_orange1, lower_orange2)
cv2.imshow('imgThreshLow', imgThreshLow)
cv2.waitKey(0)
imgThreshHigh = cv2.inRange(hsv_img, upper_orange1, upper_orange2)
cv2.imshow('imgThreshHigh', imgThreshHigh)
cv2.waitKey(0)

# Bitwise-OR low and high threshes
threshed_img = cv2.bitwise_or(imgThreshLow, imgThreshHigh)
cv2.imshow('threshed_img', threshed_img)
cv2.waitKey(0)

# smooth the image with erosion, dialation, and smooth gaussian
# first create a kernel with standard size of 5x5 pixels
kernel = np.ones((5,5),np.uint8)

# get rid of small artifacts by eroding first and then dialating 
threshed_img_smooth = cv2.erode(threshed_img, kernel, iterations = 3)
threshed_img_smooth = cv2.dilate(threshed_img_smooth, kernel, iterations = 2)

# account for cones with reflective tape by dialating first to bridge the gap between one orange edge
# and another and then erode to bring the traffic cone back to standard size
smoothed_img = cv2.dilate(threshed_img_smooth, kernel, iterations = 8)
smoothed_img = cv2.erode(smoothed_img, kernel, iterations = 7)

edges_img = cv2.Canny(smoothed_img, 100, 200)
cv2.imshow('edges_img', edges_img)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(edges_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

coordinates = []
color1 = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
tmp_obj = original_img

# Write some Text

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2
fontColor = (0, 0, 255)
lineType = 2

for cnt in contours:
    boundingRect = cv2.boundingRect(cnt)
    approx = cv2.approxPolyDP(cnt, 0.07 * cv2.arcLength(cnt, True), True)
    if len(approx) == 3:
        x, y, w, h = cv2.boundingRect(approx)
        rect = (x, y, w, h)
        cv2.rectangle(tmp_obj, (x, y), (x+w, y+h), (0, 255, 0), 3)
        bottomLeftCornerOfText = (x, y)
        cv2.putText(tmp_obj,'traffic_cone', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

cv2.imshow('Triangle extract', tmp_obj)
cv2.waitKey(0)


# contours_poly = [None]*len(contours)
# boundRect = [None]*len(contours)
# centers = [None]*len(contours)
# radius = [None]*len(contours)

# for i, c in enumerate(contours):
#     contours_poly[i] = cv2.approxPolyDP(c, 3, True)
#     boundRect[i] = cv2.boundingRect(contours_poly[i])
#     centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

# drawing = np.zeros((original_img.shape[0], original_img.shape[1], 3), dtype=np.uint8)

# for i in range(len(contours)):
#     color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
#     # cv2.drawContours(original_img, contours_poly, i, color)
#     cv2.rectangle(original_img, (int(boundRect[i][0]), int(boundRect[i][1])), \
#         (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
#     # cv2.circle(original_img, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

# cv2.imshow('Contours', original_img)
# cv2.waitKey(0)