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

[image1]: ./documentation_images/distorted_vs_undistorted.png "Distorted vs Undistorted"
[image2]: ./documentation_images/undistortedImage.jpg "Undistorted Road"
[image3]: ./documentation_images/preprocessed.jpg "Preprocessed image"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

This project contains 4 python files and they are all located in the `code/` directorium: `calibrate.py`, `preprocessing.py`, `polynomial_fit.py`,  and `lane.py`.

`preprocessing.py` and `polynomial_fit.py` are helper files and functions from there are called from the main file `lane.py`

`calibrate.py` script is separated from the rest of the project. We run it only once and it calculates the
camera calibration parameters and saves it to the `../camera_cal/calib.npz`.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

`calibrate.py` script is used to calibrate a camera, which involves determining the camera's intrinsic parameters and distortion coefficients.
#
First we define number of rows and columns in the chessboard pattern and sets the termination criteria for the corner sub-pixel refinement algorithm.
Then we prepare a grid of object points corresponding to the expected positions of corners on the chessboard. Here we assume that chessboard is fixed on the (x,y) plane at z=0. These object points are the same for all calibration images.

Then we loop through the all calibration images in the `camera_cal/` directory.
Each loaded image we convert to grayscale and attempt to find corners of the chessboard pattern using `cv2.findChessboardCorners()` function. If the corners are found, algorithm then refines their positions using the corner sub-pixel refinement algorithm.
Then we draw the detected corners on the image and briefly display the image to visually verify the calibration process.

Now we calibrate the camera using collected object and image points. The calibration results, including the camera matrix (`mtx`), distortion coefficients (`dist`), and rotation and translation vectors (`rvecs` and `tvecs`), are obtained using `cv2.calibrateCamera()`.

After that, we save the calibration result to the file mentioned above `calib.npz`.

To test this, we applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

So at first we have `lane.py` script which we run using the command `python3 lane.py`. There is bool variable `isVideo`. If the value of this variable is True, the code will process the desired video file, else if it is False, the code will process single desired image.
In both cases we call the `process()` function, where we pass the current image (or frame in case of video), `mtx` and `dist` as function arguments. `mtx` and `dist` were loaded only once, at the beggining of the program, from the mentioned `calib.npz` file.
Every other function is called from this `process()` function, and we can say this is our main function.

#### 1. Provide an example of a distortion-corrected image.

This is the example of distortion-corrected image with previously calculated camera calibration parameters:
![alt text][image2]

In the `process()` function we first call the `undistortFrame(image, width, height, mtx, dist)`. There we obtain the new camera matrix using `cv2.getOptimalNewCameraMatrix()` and then we undistort image using `cv2.undistort()`.

Undistorted images will have black indentations around the edges, so to eliminate them we crop our image using `roi` which is the one of the returing values from the `cv2.getOptimalNewCameraMatrix()` function.

So now we have our distortion-corrected image.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Before we perform a perspective transform on an image, first we need to do some preprocessing.
In the `hsv()` function we apply white and yellow mask on the undistorted image. That way we will isolate white and yellow lines on the road and filter some color irregularities or light changes.

That picture, then we pass to `canny()` funtion. There we apply some gaussian filter to eliminate some noise and to exclude false edges from consideration using `cv2.GuassianBlur()` function. Now we convert our image to binary image using `cv2.threshold()` function. And at the end we find the edges on the image using `cv2.Canny()` function.

The result is shown in the image below:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
