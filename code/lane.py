import cv2
import numpy as np
import preprocessing as prep
import polynomial_fit as fit

def process(image, mtx, dist):

    width = image.shape[1]
    height = image.shape[0]

    undistortedFrame = prep.undistortFrame(image, width, height, mtx, dist)
    hsvImage = prep.hsv(undistortedFrame)
    cannyFrame = prep.canny(hsvImage, 50, 150)
    warpedFrame, inverseM = prep.warpImage(cannyFrame)

    res = fit.lineFit(warpedFrame)

    leftFit = res['leftFit']
    rightFit = res['rightFit']
    nonzeroX = res['nonzeroX']
    nonzeroY = res['nonzeroY']
    leftLaneIndicies = res['leftLaneIndicies']
    rightLaneIndicies = res['rightLaneIndicies']

    leftCurve, rightCurve = fit.calculateCurve(leftLaneIndicies, rightLaneIndicies, nonzeroX, nonzeroY)

    bottomY = undistortedFrame.shape[0] - 1
    bottomXLeft = leftFit[0]*(bottomY**2) + leftFit[1]*bottomY + leftFit[2]
    bottomXRight = rightFit[0]*(bottomY**2) + rightFit[1]*bottomY + rightFit[2]
    vehicleOffset = undistortedFrame.shape[1]/2 - (bottomXLeft + bottomXRight)/2

    metersPerPixelX = 3.7/700
    vehicleOffset *= metersPerPixelX

    result = fit.showResult(undistortedFrame, leftFit, rightFit, inverseM, leftCurve, rightCurve, vehicleOffset)

    return result

if __name__ == "__main__":

    # Load calibrated camera parameters
    calibration = np.load('../camera_cal/calib.npz')
    mtx = calibration['mtx']
    dist = calibration['dist']

    # If 'True', video will be processed; If 'False', image will be processed
    isVideo = True

    if isVideo:
        cap = cv2.VideoCapture('../test_videos/project_video01.mp4')

        # Uncomment if want to save processed video
        # frameRate = cap.get(cv2.CAP_PROP_FPS)
        # frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # outputVideo = cv2.VideoWriter('../documentation_images_and_videos/final_result_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), frameRate, (frameWidth, frameHeight))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            result = process(frame, mtx, dist)

            # Uncomment if want to save processed video
            # outputVideo.write(result)

            cv2.imshow("result", result)
            cv2.waitKey(1)

        cap.release()
        # Uncomment if want to save processed video
        # outputVideo.release()
    else:
        image = cv2.imread("../test_images/straight_lines1.jpg")
        result = process(image, mtx, dist)
        cv2.imshow("result", result)
        cv2.waitKey(0)

    cv2.destroyAllWindows()