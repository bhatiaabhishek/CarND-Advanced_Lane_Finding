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

####3. Perspective Transform and Warping
At this stage we transform the image to a "bird-eye" view of the lane using perspective transform. The function "perspective_points" in the code returns src and dst points. I derive the src points as a trapezium whose Height, Top and Bottom widhts are specified as % of the image dimensions.

    app_lane_center = ((img_size[0]/2),img_size[1]) # This is the expected center of the lane i.e. center of the trapezium
    height_cent = 0.25 # Height of the trapezium  as % of image Y
    bott_cent = 1.0 # Bottom width as a % of image X
    top_cent = 0.41 # Top width as a % of image X

The dst points are derived by just making a trapezium into a rectangle i.e. stretch top width to be same as the bottom width.

Then functions `cv2.getPerspectiveTransform` and `cv2.warpPerspective` are used to warp the image using the `src` and `dst` points. Ideally the lanes should appear parallel in the warped image, if implemented correctly. The following couple of images show outputs of the warping stage.

<img src="https://github.com/bhatiaabhishek/CarND-Advanced_Lane_Finding/blob/master/output_images/top_down_straight_lines1.jpg" width="30%"> <img src="https://github.com/bhatiaabhishek/CarND-Advanced_Lane_Finding/blob/master/output_images/top_down_test3.jpg" width="30%"> 


####4. Finding Lane pixels and Fitting

For a given image, a rough position of the lanes can be estimated by summing up the warped image from the previous stage in the vertical direction. The python notebook submission includes histogram plots for each of the test images. The function "window_centroids" contains the code that finds lane  pixels using window-search method, given a warped image from the previous stage. In this method we split the image into levels and use sliding convolution to find out lane pixels in each level. This function returns left lane and right lane pixel coordinates for each level. In my code, I append centroids only when both the left and right lane pixels are found (convolution result > 0) in a given level.

Once the lane pixels are found, we plot those windows on the warped image as a sanity check. This gives us an idea about the window height and the width to use for a given test scenario. I found out that for a very curved road, the window height needs to be decreased to a very small value to be able to trace that lane.

An example output on a warped image is presented below:

<img src="https://github.com/bhatiaabhishek/CarND-Advanced_Lane_Finding/blob/master/output_images/top_down_windowed_test3.jpg" width="30%">

Once we have the lane pixels, these are used as data points to fit a 2nd degree polynomial using the formula below.



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
