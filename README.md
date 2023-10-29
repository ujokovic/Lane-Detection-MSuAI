**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./test_images/straight_lines1.jpg "Test image"
[image1]: ./documentation_images_and_videos/distorted_vs_undistorted.png "Distorted vs Undistorted image"
[image2]: ./documentation_images_and_videos/undistorted_image.jpg "Undistorted image"
[image3]: ./documentation_images_and_videos/preprocessed.jpg "Preprocessed image"
[image4]: ./documentation_images_and_videos/warped.jpg "Warped image"
[image5]: ./documentation_images_and_videos/histogram.jpg "Histogram of the lower half of image"
[image6]: ./documentation_images_and_videos/curve_fit.jpg "Fitted curve image"
[image7]: ./documentation_images_and_videos/radius_of_curvature.jpg "Radius of curvature formula"
[image8]: ./documentation_images_and_videos/final_result.jpg "Final result image"
[video1]: ./documentation_images_and_videos/final_result_video.avi "Final result video"

---

### Writeup / README

#### 1. Poject overview

This project contains 4 python files and they are all located in the `code/` directory: `calibrate.py`, `preprocessing.py`, `polynomial_fit.py`,  and `lane.py`.

`preprocessing.py` contains 4 functions: `undistortFrame()`, `hsv()`, `canny()`, and `warpImage()`.
`polynomial_fit.py` contains 3 functions: `lineFit()`, `calculateCurve()` and `showResult()`.
`lane.py` contains `main()` and `process()` functions.

`preprocessing.py` and `polynomial_fit.py` are helper files and functions from there are called from the main file `lane.py`

`calibrate.py` script is separated from the rest of the project. We run it only once and it calculates the
camera calibration parameters and saves it to the `../camera_cal/calib.npz`.

To demonstrate the code steps, we will use the image below:
![alt text][image0]

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

`calibrate.py` script is used to calibrate a camera, which involves determining the camera's intrinsic parameters and distortion coefficients.
#
First we define number of rows and columns in the chessboard pattern and sets the termination criteria for the corner sub-pixel refinement algorithm.
Then we prepare a grid of object points corresponding to the expected positions of corners on the chessboard. Here we assume that chessboard is fixed on the ('x','y') plane at z=0. These object points are the same for all calibration images.

Then we loop through the all calibration images in the `camera_cal/` directory.
Each loaded image we convert to grayscale and attempt to find corners of the chessboard pattern using `cv2.findChessboardCorners()` function. If the corners are found, algorithm then refines their positions using the corner sub-pixel refinement algorithm.
Then we draw the detected corners on the image and briefly display the image to visually verify the calibration process.

Now we calibrate the camera using collected object and image points. The calibration results, including the camera matrix (`mtx`), distortion coefficients (`dist`), and rotation and translation vectors (`rvecs` and `tvecs`), are obtained using `cv2.calibrateCamera()`.

After that, we save the calibration result to the file mentioned above `calib.npz`.

To test this, we applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

So at first we have `lane.py` script which we run using the command `python3 lane.py` (**Note that it is important to run this script from the `code/` directory**). There is bool variable `isVideo`. If the value of this variable is True, the code will process the desired video file, else if it is False, the code will process single desired image.
In both cases we call the `process()` function, where we pass the current image (or frame in case of video), `mtx` and `dist` as function arguments. `mtx` and `dist` were loaded only once, at the beggining of the program, from the mentioned `calib.npz` file.
Every other function is called from this `process()` function, and we can say this is our main function.

#### 1. Provide an example of a distortion-corrected image.

This is the example of distortion-corrected image with previously calculated camera calibration parameters:
![alt text][image2]

In the `process()` function we first call the `undistortFrame()`. There we obtain the new camera matrix using `cv2.getOptimalNewCameraMatrix()` and then we undistort image using `cv2.undistort()`.

Now we have our distortion-corrected image.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Before we perform a perspective transform on an image, first we need to do some preprocessing.
In the `hsv()` function we apply white and yellow mask on the undistorted image. That way we will isolate white and yellow lines on the road and filter some color irregularities or light changes.

That picture, then we pass to `canny()` funtion. There we apply some gaussian filter to eliminate some noise and to exclude false edges from consideration using `cv2.GuassianBlur()` function. Now we convert our image to binary image using `cv2.threshold()` function. And at the end we find the edges on the image using `cv2.Canny()` function.

The result is shown in the image below:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

After we have successfuly identified edges on our image, now we need to do a perspective transform. In other words, effectively we find a region of interest where we are going to apply alogithm for curve fitting and lane finding.

So after the `canny()` function, we call the `warpImage()` function.
First we need to do a perspective transform calling `cv2.getPerspectiveTransform()` function, which takes source (`src`) and destination (`dst`) points, as an inputs. `src` points are the coordinates of four points in the original image, and `dst` points are the corresponding coordinates to which you want to map the source points. We don't exactly hardcode the points, but based on the position of the car camera, we can experimentally determine the percentage of the image's width and height occupied by the road lane. So the code looks like this:

###### Source points
    bottom_left = [round(width * 0.15), round(height * 0.9)]
    bottom_right = [round(width * 0.9), round(height * 0.9)]
    top_left = [round(width * 0.43), round(height * 0.65)]
    top_right = [round(width * 0.59), round(height * 0.65)]

###### Destination points
    bottom_left = [0, height]
    bottom_right = [width, height]
    top_left = [0, height * 0.25]
    top_right = [width, height * 0.25]

Here we can see for example the bottom_left source point is located at the 15% of image's width and at 90% of image's height.

`cv2.getPerspectiveTransform(src, dst)` gives us transofrmation matrix `M`, and after that we call `cv2.getPerspectiveTransform(dst, src)`
to calculate inverse matrix `inverseM`. Inverse matrix, we will later use to unwarp the processed image.

Then finally we call `cv2.warpPerspective()` to warp an image. Function `warpImage()` returns the warped image and the inverse matrix `M`.

The warped image, can be seen below:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The `lineFit()` function is a key component in the process of lane line detection and polynomial fitting within images.
We can say that this function consists of 5 important parts which are listed below:

1. **Histogram Computation** - Function starts by calculating a histogram of the lower half of the input binary image. The histogram reveals the distribution of pixel intensities along the horizontal axis and is used to identify initial estimates of the left and right lane positions. The result is shown in the picture below:

   ![alt text][image5]

2. **Lane Detection Windows** - Then we use 'window-based' approach, dividing the image into a specified number of vertical windows (typically 9 windows). A sliding window search is conducted to locate lane pixels within each window.

3. **Lane Pixel Extraction** - The 'x' and 'y' coordinates of non-zero pixels are collected within each window. Variables `leftLaneIndicies` and `rightLaneIndicies` are filled those coordinates.

4. **Polynomial Curve Fitting** - With the collected pixel coordinates, the function conducts polynomial curve fitting. It fits second-degree polynomial curves to represent the left and right lane lines. The coefficients of these polynomial equations are stored in variables `leftFit` and `rightFit`.

5. **Return value** - The function returns a dictionary named `ret`, containing essential results:
   - `leftFit`: The coefficients of the polynomial fit for the left lane line.
   - `rightFit`: The coefficients of the polynomial fit for the right lane line.
   - `nonzeroX`: The 'x' coordinates of non-zero (non-black) pixels in the image.
   - `nonzeroY`: The 'y' coordinates of non-zero pixels in the image.
   - `leftLaneIndicies`: The indices of pixels belonging to the left lane.
   - `rightLaneIndicies`: The indices of pixels belonging to the right lane.

After successfuly fitting a curve, we get the result below:

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Now, to determine radius of curvature for the left and right lane we use the function `calculateCurve()`.
This function calculates the curvature of the lane lines based on pixel coordinates and converts the results into real world measurements.

The function takes as input the pixel coordinates of the left and right lane lines (`leftLaneIndicies` and `rightLaneIndicies`) and the 'x' and 'y' coordinates of all non-zero pixels in the image (`nonzeroX` and `nonzeroY`)

We can say that this function consists of 5 important parts which are listed below:

1. **World Space Conversion** -  To convert the pixel coordinates into real world units, we define conversions for both the 'y' and 'x' dimensions. `metersPerPixelY` represents the conversion from pixels to meters in the 'y' dimension, and `metersPerPixelX` represents the conversion in the 'x' dimension.

2. **Pixel Extraction** -  Then we extract the 'x' and 'y' coordinates of the left and right lane pixels based on the provided indices.

3. **Polynomial Curve Fitting** -  Now, the new second-degree polynomial curves are fitted to the extracted pixel coordinates in world space for both the left and right lane lines. These curves are stored in the `leftFitCurve` and `rightFitCurve` variables.

4. **Radius of Curvature Calculation** - The radius of curvature for each lane line is calculated using the fitted curves and the formula below:

    ![alt text][image7]

    The results, `leftCurveRadius` and `rightCurveRadius` are expressed in meters. It is possible to choose any 'y' value of the image to compute the radius of curvature, but it is the best to use the 'y' closest to the vehicle.

5. **Return value** - The function returns the calculated radii of curvature for the left and right lane lines in meters.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Function that shows the final result with lane detected, radii of curvature for the left and right lane line and the vehicle offset from lane center is implemented in the `showResult()` function.

We can say that this function consists of 8 important parts which are listed below:

1. **Generate Plotting Values**: First we generate a set of 'x' and 'y' values for plotting the detected lane lines based on the provided polynomial coefficients (`leftFit` and `rightFit`). It calculates the lane line pixel positions in the image for the entire vertical range, resulting in `leftFitx` and `rightFitx` arrays.

2. **Create ColorWarp Image**: Now we create a blank image `colorWarp` to serve as a canvas for drawing the detected lane lines. This image has dimensions matching the original undistorted frame, and it is initialized with zeros, creating a black canvas.

3. **Recast Points for Drawing**: The 'x' and 'y' coordinates of the detected left and right lane lines are recast into a format suitable for use with the `cv2.fillPoly()` function. These points are stored in arrays named `ptsLeft` and `ptsRight`, and then combined into a single `pts` array.

4. **Draw Lane Lines**: The detected lane lines are drawn onto the `colorWarp` image using `cv2.fillPoly()`. This results in two highlighted lane regions on the black canvas.

5. **Inverse Perspective Transform**: The black canvas with the highlighted lane regions is transformed from the bird's-eye view back to the original perspective using the provided inverse perspective matrix (`inverseM` - we've calculated it before in the `warpImage()` function). The result is an image that aligns with the original undistorted frame's perspective.

6. **Combine Images**: The transformed image with the annotated lane regions is combined with the original undistorted frame using the `cv2.addWeighted()` function.

7. **Lane Curvature Annotation**: Then we add the text annotations to the result image. It displays the radius of curvature (which was calculated as a mean value of the `leftCurve` `rightCurve` values) of the lane lines and the vehicle's offset from the lane center.

8. **Return value**: The function returns the final annotated image.

The final result can be seen in the image below:
![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Final result video was made from `test_videos/project_video01.mp4`
Here's a [link to my video result](./documentation_images_and_videos/final_result_video.avi).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the issues occured when i run challenge03.mp4 video. Video runs for a little bit then just crashes and the program returns the following error:
**'in polyfit raise TypeError("expected non-empty vector for x")'**. So thats the one thing i know for now that needs to be fixed.
One of the potential improvements could involve implementing a more sophisticated algorithm for preprocessing the image. For instance, in challenge03.mp4, we can see that the lighting conditions are causing significant issues, which hinder our ability to detect the lanes.
That's all I can think of for now.