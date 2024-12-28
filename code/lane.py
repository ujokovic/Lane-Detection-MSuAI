import cv2
import numpy as np
import preprocessing as prep
import polynomial_fit as fit

def process(image):
    hsvImage = prep.hsv(image)
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

    bottomY = image.shape[0] - 1
    bottomXLeft = leftFit[0]*(bottomY**2) + leftFit[1]*bottomY + leftFit[2]
    bottomXRight = rightFit[0]*(bottomY**2) + rightFit[1]*bottomY + rightFit[2]
    vehicleOffset = image.shape[1]/2 - (bottomXLeft + bottomXRight)/2

    metersPerPixelX = 3.7/700
    vehicleOffset *= metersPerPixelX

    result = fit.showResult(image, leftFit, rightFit, inverseM, leftCurve, rightCurve, vehicleOffset)

    return result

if __name__ == "__main__":
    # Options: Set to True for video, image, or camera
    isVideo = True  # Set to True to process video
    isImage = False  # Set to True to process static image
    useCamera = False  # Set to True to process from camera

    if isVideo:
        cap = cv2.VideoCapture('../test_videos/challenge03.mp4')

        # Uncomment if want to save processed video
        # frameRate = cap.get(cv2.CAP_PROP_FPS)
        # frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # outputVideo = cv2.VideoWriter('../documentation_images_and_videos/final_result_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), frameRate, (frameWidth, frameHeight))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            result = process(frame)

            # Uncomment if want to save processed video
            # outputVideo.write(result)

            cv2.imshow("result", result)
            cv2.waitKey(1)

        cap.release()
        # Uncomment if want to save processed video
        # outputVideo.release()

    elif isImage:
        image = cv2.imread("../test_images/straight_lines1.jpg")
        result = process(image)
        cv2.imshow("result", result)
        cv2.waitKey(0)

    elif useCamera:
        # Open camera (0 is usually the default camera)
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            result = process(frame)
            cv2.imshow("result", result)

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    cv2.destroyAllWindows()
