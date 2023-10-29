import cv2
import numpy as np

def undistortFrame(image, width, height, mtx, dist):

    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 0, (width, height))
    undistortedImg = cv2.undistort(image, mtx, dist, None, newCameraMtx)

    return undistortedImg

def hsv(image):

    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Mask for white color
    lower = np.array([0, 0, 200])
    upper = np.array([255, 30, 255])
    maskW = cv2.inRange(hsvImage, lower, upper)

    # Mask for yellow color
    lower = np.array([10, 100, 100])
    upper = np.array([30, 255, 255])
    maskY = cv2.inRange(hsvImage, lower, upper)

    mask = cv2.bitwise_xor(maskW, maskY)

    filteredImage = cv2.bitwise_and(image, image, mask = mask)

    return filteredImage

def canny(image, thr1, thr2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 125, 225, cv2.THRESH_BINARY)
    canny = cv2.Canny(thresh, thr1, thr2)
    return canny

def warpImage(image):

    width = image.shape[1]
    height = image.shape[0]

    # Source points
    bottom_left = [round(width * 0.15), round(height * 0.9)]
    bottom_right = [round(width * 0.9), round(height * 0.9)]
    top_left = [round(width * 0.43), round(height * 0.65)]
    top_right = [round(width * 0.59), round(height * 0.65)]

    src = np.float32([bottom_left, bottom_right, top_right, top_left])

    # Uncomment if want to show source points on the image
    # cv2.circle(image, bottom_left, 10, (0,0,255), -1)
    # cv2.circle(image, bottom_right, 10, (0,0,255), -1)
    # cv2.circle(image, top_left, 10, (0,0,255), -1)
    # cv2.circle(image, top_right, 10, (0,0,255), -1)
    # cv2.imshow("points", image)
    # cv2.waitKey(0)

    # Destination points
    bottom_left = [0, height]
    bottom_right = [width, height]
    top_left = [0, height * 0.25]
    top_right = [width, height * 0.25]

    dst = np.float32([bottom_left, bottom_right, top_right, top_left])

    M = cv2.getPerspectiveTransform(src, dst)
    inverseM = cv2.getPerspectiveTransform(dst, src)
    warpedImage = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_NEAREST)

    return warpedImage, inverseM