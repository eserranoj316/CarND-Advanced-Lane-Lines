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

[image1]: ./output_images/originalImage_undistortedImage.png "Undistorted"
[image2]: ./output_images/originalTestImage_undistortedTestImage.png "Road Transformed"
[image3]: ./output_images/undistortedTestImage_SColorThreshold.png "S Color threshold Example"
[image4]: ./output_images/undistortedTestImage_SobelXThreshold.png "SobelX gradient thresholding"
[image5]: ./output_images/undistortedTestImage_SobelX-SColorThreshold.png "Combination of S and SobelX Gradient"
[image6]: ./output_images/straight_lines1_undistorts_perspectiveTransform.png "Perspective Transform1"
[image7]: ./output_images/straight_lines2_undistorts_perspectiveTransform.png "Perspective Transform2"
[image8]: ./output_images/SobelX-SColorThreshold_PerspectiveTransform.png "Perspective Transform3"
[image9]: ./output_images/binary_warped_birdseyeview.png "BirdsEyeView"
[image10]: ./output_images/histogram.png "BirdsEyeView_histogram"
[image11]: ./output_images/originalImage_detectedlane_radiuscurv_offset.png "lane detected_radius curvature_car position"
[video1]: ./project4_AnnotatedVideo.mp4 "Video"

---
###Writeup / README

###Camera Calibration


The code for calibrating camera is in calibrate_camera function (P4AdvancedLaneLines.py) which requires following input arguments: globpattern,nx,ny,image_shape. The function reads in all calibration images one by one, convert to gray scale and using  cv2.findChessboardCorners to search for inner corners (intersection of two blaclk squares) and append them in imgpoints list. A corresponding object point is then appended to objpoints list. imgpoints and objpoints are then used to calibrate the camera via cv2.calibrateCamera. Function returns cameraMatrix(mtx),  distortion coefficients(dist), rotation vectors(rvecs), and translation vectors(tvecs).
mtx and dist will be used later on in cv2.undistort function. See below picture showing original and undistorted image.
![alt text][image1]

###Pipeline (single images)

####1. Applying cv2.undistort to test image using the camera matrix and distortion coefficients obtained from cv2.calibrateCamera

image = mpimg.imread("test_images/test5.jpg")
undist = cv2.undistort(image, mtx, dist, None, mtx) 
![alt text][image2]
####2. Applying color transforms and gradients to create a thresholded binary image.
I used combination of SobelX gradient and Saturation(S) channel (HLS space) thresholding to create thresholded binary image (P4AdvancedLaneLines.py line 316). For S thresholding,  I set ( s_thresh_min, s_thresh_max) to (170,255).
The resulting s_binary image is:
![alt text][image3]
For SobelX gradient thresholding, I set (thresh_min,thresh_max) to (20,100). The code for sobelx gradient is inside abs_sobel_thresh function. Executing this function with kernel size of 9 and the mentioned (thresh_min,thresh_max) settings above will produced a gradx binary image as shown:
![alt text][image4]
Combining these two binary images (gradx and s_binary) will give the final binary image:

combined = np.zeros_like(s_binary)
combined[(s_binary == 1) | (gradx == 1)] = 1
    
![alt text][image5]


####3. Performing a perspective transform of a transformed image.

By trying several combinations of four source and destination points  and checking which combination will produce a straight and vertical lane lines for the perspective transformation (P4AdvancedLaneLines.py line 338) of the  straight_lines1.jpg and straight_lines2.jpg, source and destinations points obtained were:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 600, 444      | 320, 0        | 
| 206, 720      | 320, 720      |
| 1120, 720     | 960, 720      |
| 675, 444      | 960, 0        |

Using these source ad destination points I calculated the Matrix and Inverse Matrix transform parameters. 

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

I verified that the perspective transform is working properly by obtaining a warped image showing that the lane lines are almost straight and vertical (See image below showing original test images and the corresponding warped images with red lines drawn on 4 source and destination points) 

![alt text][image6]
![alt text][image7]


####4. Identifying lane-line pixels and fit their positions with a polynomial
Applying perpective transform on the thresholded binary test image yielded the following:
![alt text][image8]


##Line Finding Method: Peaks in a Histogram
To find the left and right lane-line pixels, we need to locate first the best x-position of the base lines. This is done by getting the histogram 
along all the columns in the lower half of the image shown below:
![alt text][image9]

histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
plt.plot(histogram)

Resulting histogram is:
![alt text][image10]

Finding the peak of the left and right halves of the histogram will give us the starting base point for the left and right lines.

midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

###Implement Sliding Windows and Fit a Polynomial 

The sliding_window function shown below is use to find the left and right lane-line pixels and perform a second order polynomial fit on pixels positions to obtain the left_fit and right_fit polynomial coefficients. 

``` python

def sliding_window(combined_threshold,image):
    global mtx, dist, M, Minv, left_fit, right_fit
    binary_warped = cv2.warpPerspective(combined_threshold, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)   
    undist = cv2.undistort(image, mtx, dist, None, mtx) 
    #Line Finding Method: Peaks in a Histogram
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    #Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 9
    window_height = np.int(binary_warped.shape[0]/nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (orgimage.shape[1], orgimage.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    plt.imshow(result)
    
 ```
 
The pipelines function code is shown below. It accepts original image as an input. Image can be from a single image file or a single frame from a video. The function applies combination of Sobelx and S color (HLS space) thresholding to the image and perform a perspective transform using the previous calculated Matrix(M) transform parameters to obtain the thresholded binary warped image. By using the left_fit and right_fit polynomial coefficients obtained from executing the sliding_window function the corresponding left and right lane-line pixels of the given image are detected. A second order polynomial is fitted on the new extracted left and right line pixels position.

``` python

def pipelines(image):
    global mtx, dist, M, Minv, left_fit, right_fit
    ksize = 9 # Choose a larger odd number to smooth gradient measurements
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    # Apply each of the thresholding functions
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    #combining thresholds 
    combined = np.zeros_like(s_binary)
    combined[(s_binary == 1) | (gradx == 1)] = 1
    binary_warped = cv2.warpPerspective(combined, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

```


####5. Calculating the radius of curvature of the lane and the position of the vehicle with respect to center.
The function calcRadiusCurvature calculates the radius of curvature using the formula for Rcurve evaluating over y value corresponding to the bottom of the image:
```python

#f(y)=Ay​2​​+By+C - 
#R​curve​​=​​​((1+(2Ay+B)^​2​​)^​3/2)/|2A|
def calcRadiusCurvature(ploty,left_fitx,right_fitx):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    #maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    curvature = (left_curverad + right_curverad) / 2.0
    result = "Radius of Curvature = " +  str(int(curvature)) + '(m)'
    return result 
```
---
The position of the vehicle with respect to center is calcuted via calc_offset function.

```python 
def calc_offset(org_image, leftx, rightx):
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    image_middle_pos = org_image.shape[1] / 2
    left_lane_pos = leftx[0]    # x position for left lane
    right_lane_pos = rightx[0]  # x position for right lane
    car_middle_pos = int((right_lane_pos + left_lane_pos) / 2)
    pixels_offset = image_middle_pos - car_middle_pos
    meters_offset = round(xm_per_pix * pixels_offset, 2)
    if meters_offset >= 0:
        result = "Vehicle is " + str(meters_offset) + "m left of center"
    else:
        result = "Vehicle is " + str(abs(meters_offset)) + "m right of center"
    return result
```

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

calcRadiusCurvature and calc_offset functions are being called inside the pipeline function. Here is an example of the pipeline result applied on a test image:

![alt text][image11]

---

###Pipeline (video)
Here's a [link to video after running through the pipeline](./project4_AnnotatedVideo.mp4)

---

###Discussion
By looking at the final video the lane markings shows problem on area where there are shadows. Fine tuning the thresholding and gradient parameters or add additional thresholding/gradient techniques other than SobelX gradient and S channel (HLS space) might improved the lane detection further.


