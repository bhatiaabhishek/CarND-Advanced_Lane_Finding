# CarND-Advanced_Lane_Finding



**Advanced Lane Finding Project**

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

[image1]: ./output_images/calibration5.jpg "Distorted"
[image2]: ./output_images/undist_calibration5.jpg "Undistorted"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


###Camera Calibration

In the first stage of the pipeline, the image needs to be undistorted i.e. camera distortion needs to be removed. To be able to do that, we need to calculate camera matrix and distortion coefficients for the camera first.
These parameters are calculated using chessboard images from the same camera (provided in ./camera_cal). The code for camera calibration is in the first code cell of carnd-adv_lane_finding.ipynb. A class called "calibration_mat" is used to store all the calibration parameters.

class calibration_mat:
    def __init__(self,ret,mtx,dist,rvecs,tvecs):
        self.ret = ret
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        

First, "object points" are created, which are 3D (x, y, z) coordinates of the chessboard corners in the real world. It is assumed that the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time we successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. The corners in the image plane can be found using function `cv2.findChessboardCorners(gray, (nx, ny), None)`.

The camera calibration and distortion coefficients are calculated using the `cv2.calibrateCamera()` function. The calculated parameters (mtx, dist) are then used to undistort the chessboard images to verify correctness. The `cv2.undistort()` function is used for this purporse. The following image shows an undistorted (right) chessboard image from this project.

<img src="https://github.com/bhatiaabhishek/CarND-Advanced_Lane_Finding/blob/master/camera_cal/calibration5.jpg" width="30%"> <img src="https://github.com/bhatiaabhishek/CarND-Advanced_Lane_Finding/blob/master/output_images/undist_calibration5.jpg" width="30%">


###Pipeline (single images)

My pipeline is basically the function called "find_lane" in the code. This function calls other functions to implement different stages of the pipeline.

####1. Distortion Correction
The first stage of my pipeline is to undistort each frame. The distortion matrices calculated above (camera calibration) are used to undistort each image. The following is the output using one of the test images (undistorted image on the right).

<img src="https://github.com/bhatiaabhishek/CarND-Advanced_Lane_Finding/blob/master/test_images/test5.jpg" width="30%"> <img src="https://github.com/bhatiaabhishek/CarND-Advanced_Lane_Finding/blob/master/output_images/undist_test5.jpg" width="30%">

####2. Color and Gradient Threshold 
The function "color_n_grad_thresh" calls the required functions to create a thresholded binary image. A combination of S (from HLS) and V (from HSV) channels are used for color thresholding. For gradient thresholding, a combination of absolute (X and Y), magnitude and direction gradients are used.

    mag_binary = grad_mag_thrsh(img,sobel_kernel,(mag_thresh_min,mag_thresh_max))
    dir_binary = grad_dir_thrsh(img,sobel_kernel,(dir_thresh_min,dir_thresh_max))
    absX_binary = grad_abs_thrsh(img,sobel_kernel,'x',(absX_thresh_min,absX_thresh_max))
    absY_binary = grad_abs_thrsh(img,sobel_kernel,'y',(absY_thresh_min,absY_thresh_max))

The following images show how each of the gradients affect the test image.

Original Image: 
<img src="https://github.com/bhatiaabhishek/CarND-Advanced_Lane_Finding/blob/master/test_images/straight_lines1.jpg" width="30%"> 

Color thresholding : 
<img src="https://github.com/bhatiaabhishek/CarND-Advanced_Lane_Finding/blob/master/output_images/color_thresh_straight_lines1.jpg" width="30%">

Gradient thresholding : 
<img src="https://github.com/bhatiaabhishek/CarND-Advanced_Lane_Finding/blob/master/output_images/grad_thresh_straight_lines1.jpg" width="30%">

Final Binary Image: 
<img src="https://github.com/bhatiaabhishek/CarND-Advanced_Lane_Finding/blob/master/output_images/binary_straight_lines1.jpg" width="30%">

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
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

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
