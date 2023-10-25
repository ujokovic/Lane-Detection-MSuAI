import cv2
import numpy as np
import matplotlib.pyplot as plt

def lineFit(image):

    height = image.shape[0]
    # Take a histogram of the bottom half of the image
    histogram = np.sum(image[int(height/2):,:], axis=0)

    # Show the calculated histogram
    # plt.plot(histogram)
    # plt.show()

    # Create an output image to draw on and visualize the result
    outImage = (np.dstack((image, image, image))*255).astype('uint8')

    # Half of the x-axis
    midpoint = int(histogram.shape[0]/2)

    leftPeakXBase = np.argmax(histogram[:midpoint])
    rightPeakXBase = np.argmax(histogram[midpoint:]) + midpoint

    leftPeakXCurrent = leftPeakXBase
    rightPeakXCurrent = rightPeakXBase

    # Window setup
    numOfWindows = 9
    windowHeight = int(height/numOfWindows)
    windowMargin = 100
    minpix = 50

    # Find the coordinates of all nonzero pixels in the image
    nonzero = image.nonzero()
    nonzeroY = np.array(nonzero[0])
    nonzeroX = np.array(nonzero[1])

    leftLaneIndicies = []
    rightLaneIndicies = []

    for window in range(numOfWindows):
        # Calculate window boundaries
        windowYLow = height - (window+1)*windowHeight
        windowYHigh = height - window*windowHeight
        windowXLeftLow = leftPeakXCurrent - windowMargin
        windowXLeftHigh = leftPeakXCurrent + windowMargin
        windowXRightLow = rightPeakXCurrent - windowMargin
        windowXRightHigh = rightPeakXCurrent + windowMargin

        # Draw the windows on the visualization image
        cv2.rectangle(outImage,(windowXLeftLow,windowYLow),(windowXLeftHigh,windowYHigh),(0,255,0), 2)
        cv2.rectangle(outImage,(windowXRightLow,windowYLow),(windowXRightHigh,windowYHigh),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window
        goodLeftIndicies = ((nonzeroY >= windowYLow) & (nonzeroY < windowYHigh) & (nonzeroX >= windowXLeftLow) & (nonzeroX < windowXLeftHigh)).nonzero()[0]
        goodRightIndicies = ((nonzeroY >= windowYLow) & (nonzeroY < windowYHigh) & (nonzeroX >= windowXRightLow) & (nonzeroX < windowXRightHigh)).nonzero()[0]

        leftLaneIndicies.append(goodLeftIndicies)
        rightLaneIndicies.append(goodRightIndicies)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(goodLeftIndicies) > minpix:
            leftPeakXCurrent = int(np.mean(nonzeroX[goodLeftIndicies]))
        if len(goodRightIndicies) > minpix:
            rightPeakXCurrent = int(np.mean(nonzeroX[goodRightIndicies]))

    leftLaneIndicies = np.concatenate(leftLaneIndicies)
    rightLaneIndicies = np.concatenate(rightLaneIndicies)

    # Extract left and right line pixel positions
    leftX = nonzeroX[leftLaneIndicies]
    leftY = nonzeroY[leftLaneIndicies]
    rightX = nonzeroX[rightLaneIndicies]
    rightY = nonzeroY[rightLaneIndicies]

    leftFit = np.polyfit(leftY, leftX, 2)
    rightFit = np.polyfit(rightY, rightX, 2)

    ret = {}
    ret['leftFit'] = leftFit
    ret['rightFit'] = rightFit
    ret['nonzeroX'] = nonzeroX
    ret['nonzeroY'] = nonzeroY
    ret['outImage'] = outImage
    ret['leftLaneIndicies'] = leftLaneIndicies
    ret['rightLaneIndicies'] = rightLaneIndicies

    return ret

def calculateCurve(leftLaneIndicies, rightLaneIndicies, nonzeroX, nonzeroY):
	"""
	Calculate radius of curvature in meters
	"""
	yEval = 719  # 720p video/image, so last (lowest on screen) y index is 719

	# Define conversions in x and y from pixels space to meters
	metersPerPixelY = 30/720 # meters per pixel in y dimension
	metersPerPixelX = 3.7/700 # meters per pixel in x dimension

	# Extract left and right line pixel positions
	leftX = nonzeroX[leftLaneIndicies]
	leftY = nonzeroY[leftLaneIndicies]
	rightX = nonzeroX[rightLaneIndicies]
	rightY = nonzeroY[rightLaneIndicies]

	# Fit new polynomials to x,y in world space
	leftFitCurve = np.polyfit(leftY*metersPerPixelY, leftX*metersPerPixelX, 2)
	rightFitCurve = np.polyfit(rightY*metersPerPixelY, rightX*metersPerPixelX, 2)
	# Calculate the new radii of curvature
	leftCurveRadius = ((1 + (2*leftFitCurve[0]*yEval*metersPerPixelY + leftFitCurve[1])**2)**1.5) / np.absolute(2*leftFitCurve[0])
	rightCurveRadius = ((1 + (2*rightFitCurve[0]*yEval*metersPerPixelY + rightFitCurve[1])**2)**1.5) / np.absolute(2*rightFitCurve[0])
	# Now our radius of curvature is in meters

	return leftCurveRadius, rightCurveRadius

def visualize(image, ret):
	"""
	Visualize the predicted lane lines with margin, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	leftFit = ret['leftFit']
	rightFit = ret['rightFit']
	nonzeroX = ret['nonzeroX']
	nonzeroY = ret['nonzeroY']
	leftLaneIndicies = ret['leftLaneIndicies']
	rightLaneIndicies = ret['rightLaneIndicies']

	# Create an image to draw on and an image to show the selection window
	outImage = (np.dstack((image, image, image))*255).astype('uint8')
	windowImage = np.zeros_like(outImage)
	# Color in left and right line pixels
	outImage[nonzeroY[leftLaneIndicies], nonzeroX[leftLaneIndicies]] = [255, 0, 0]
	outImage[nonzeroY[rightLaneIndicies], nonzeroX[rightLaneIndicies]] = [0, 0, 255]

	# Generate x and y values for plotting
	plotY = np.linspace(0, image.shape[0]-1, image.shape[0])
	leftFitX = leftFit[0]*plotY**2 + leftFit[1]*plotY + leftFit[2]
	rightFitX = rightFit[0]*plotY**2 + rightFit[1]*plotY + rightFit[2]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	margin = 100  # NOTE: Keep this in sync with linefit()
	leftLineWindow1 = np.array([np.transpose(np.vstack([leftFitX-margin, plotY]))])
	leftLineWindow2 = np.array([np.flipud(np.transpose(np.vstack([leftFitX+margin, plotY])))])
	leftLinePts = np.hstack((leftLineWindow1, leftLineWindow2))
	rightLineWindow1 = np.array([np.transpose(np.vstack([rightFitX-margin, plotY]))])
	rightLineWindow2 = np.array([np.flipud(np.transpose(np.vstack([rightFitX+margin, plotY])))])
	rightLinePts = np.hstack((rightLineWindow1, rightLineWindow2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(windowImage, np.int_([leftLinePts]), (0,255, 0))
	cv2.fillPoly(windowImage, np.int_([rightLinePts]), (0,255, 0))
	result = cv2.addWeighted(outImage, 1, windowImage, 0.3, 0)
	plt.imshow(result)
	plt.plot(leftFitX, plotY, color='yellow')
	plt.plot(rightFitX, plotY, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	plt.show()

	plt.gcf().clear()


def finalVisualization(undistortedFrame, leftFit, rightFit, inverseM, leftCurve, rightCurve, vehicleOffset):
	"""
	Final lane line prediction visualized and overlayed on top of original image
	"""
	# Generate x and y values for plotting
	plotY = np.linspace(0, undistortedFrame.shape[0]-1, undistortedFrame.shape[0])
	leftFitx = leftFit[0]*plotY**2 + leftFit[1]*plotY + leftFit[2]
	rightFitx = rightFit[0]*plotY**2 + rightFit[1]*plotY + rightFit[2]

	# Create an image to draw the lines on
	#warp_zero = np.zeros_like(warped).astype(np.uint8)
	#colorWarp = np.dstack((warp_zero, warp_zero, warp_zero))
	colorWarp = np.zeros((720, 1280, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions

	# Recast the x and y points into usable format for cv2.fillPoly()
	ptsLeft = np.array([np.transpose(np.vstack([leftFitx, plotY]))])
	ptsRight = np.array([np.flipud(np.transpose(np.vstack([rightFitx, plotY])))])
	pts = np.hstack((ptsLeft, ptsRight))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(colorWarp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	unwarpedImage = cv2.warpPerspective(colorWarp, inverseM, (undistortedFrame.shape[1], undistortedFrame.shape[0]))
	# Combine the result with the original image
	result = cv2.addWeighted(undistortedFrame, 1, unwarpedImage, 0.3, 0)

	# Annotate lane curvature values and vehicle offset from center
	averageCurve = (leftCurve + rightCurve)/2
	label = 'Radius of curvature: %.1f m' % averageCurve
	result = cv2.putText(result, label, (30,40), 0, 1, (0,0,0), 2, cv2.LINE_AA)

	label = 'Vehicle offset from lane center: %.1f m' % vehicleOffset
	result = cv2.putText(result, label, (30,70), 0, 1, (0,0,0), 2, cv2.LINE_AA)

	return result

