import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import sys

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

def calibrate_camera(globpattern,nx,ny,image_shape):
    images = glob.glob(globpattern)
    #3d points in real world space
    objpoints = []
     #2D points in image plane
    imgpoints = []
    #create object point (0,0,0), (1,0,0), (2,0,0), .....(7,5,0)
    objp = np.zeros((ny*nx,3),np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    for fname in images:
        #read each image
        img = mpimg.imread(fname)
        #convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
    #get image size on what of the calibration images.
    img = mpimg.imread('camera_cal/calibration1.jpg')
    #calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (image_shape[1],image_shape[0]) ,None,None) 
    return (ret, mtx, dist, rvecs, tvecs)


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1 , 0,ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0 , 1,ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1        
    # 6) Return this mask as your binary_output image
    return sxbinary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1 , 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0 , 1,ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    abs_sobel = pow((pow(sobelx,2.0) + pow(sobely,2.0)),0.5)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1 
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1 , 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0 , 1,ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_dir = np.arctan2(abs_sobely, abs_sobelx) 
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(grad_dir)
    # 6) Return this mask as your binary_output image
    dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    return dir_binary


def color_thresh(image, color_thresh=(0,255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = np.zeros_like(gray)
    binary[(gray > color_thresh[0]) & (gray <= color_thresh[1])] = 1
    return binary

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
    offset = calc_offset(image, leftx, rightx)
    radiusCurvature = calcRadiusCurvature(ploty,left_fitx,right_fitx)
    #cv2.putText(result,radiusCurvature , (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    cv2.putText(result,radiusCurvature , (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),4)
    cv2.putText(result,offset , (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),4)
    return result



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

def showPlotBeforeAfter(img1,img2,title1,title2,cmap=None,axis='on'):
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    a.axis(axis)
    a.set_title(title1)
    plt.imshow(img1, cmap=cmap)    
    a=fig.add_subplot(1,2,2)
    a.axis(axis)
    a.set_title(title2)
    plt.imshow(img2,cmap=cmap)

def process_image(image):
    result = pipelines(image)
    return result
def calcRadiusCurvature(ploty,left_fitx,right_fitx):
    # Define conversions in x and y from pixels space to meters
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



if __name__ == "__main__":
    #list of chess boards to use for calibration
    globpattern = "camera_cal/calibration*.jpg"  
    #chess board number of horizontal and vertical inner corners (intersection between two black square
    nx,ny = (9,6) 
    
    image = mpimg.imread('camera_cal/calibration1.jpg')
    #1.calibrating camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(globpattern,nx,ny,image.shape)
    
    undist = cv2.undistort(image, mtx, dist, None, mtx) 
    showPlotBeforeAfter(image,undist,"original image","undistorted image")
    
    #applying the cv2.undistort to test image using the camera matrix and distortion coefficients
    #obtained from cv2.calibrateCamera
    image = mpimg.imread("test_images/test5.jpg")
    undist = cv2.undistort(image, mtx, dist, None, mtx) 
    showPlotBeforeAfter(image,undist,"original image","undistorted image")
    
    
    #Pipeline (single images)
    #2. Combination of Sobelx and S channel (from HLS space) Thresholding 
    # Choose a Sobel kernel size
    ksize = 9 # Choose a larger odd number to smooth gradient measurements
    image = mpimg.imread("test_images/test5.jpg")
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
    showPlotBeforeAfter(undist,s_binary,"undistorted test image","S Color Thresholding", cmap='gray')
    
    
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    showPlotBeforeAfter(undist,gradx,"undistorted test image","SobelX Gradient", cmap='gray')
    
       
    #combining thresholds 
    combined = np.zeros_like(s_binary)
    combined[(s_binary == 1) | (gradx == 1)] = 1
    showPlotBeforeAfter(undist,combined,"undistorted test image","Combination of S and SobelX Gradient", cmap='gray')

    #by trial and error. Searching for good matching source and destination points
    #that will give us a straight and vertical lane lines from bird's eye view after 
    #executing perspective transform on the original image.  
    src = np.float32([(600,444), (206,720), (1120,720),(675,444)])
    dst = np.float32([(320,0), (320,720), (960,720),(960,0)])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
     
    image = mpimg.imread("test_images/straight_lines1.jpg")
    #image = mpimg.imread("test_images/straight_lines2.jpg")
    undist = cv2.undistort(image, mtx, dist, None, mtx) 
    warped = cv2.warpPerspective(undist, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    #plt.imshow(warped)
    #draw lines in undistort original image
    cv2.line(undist,tuple(src[0]),tuple(src[1]),(255,0,0),5)
    cv2.line(undist,tuple(src[1]),tuple(src[2]),(255,0,0),5)
    cv2.line(undist,tuple(src[2]),tuple(src[3]),(255,0,0),5)
    cv2.line(undist,tuple(src[3]),tuple(src[0]),(255,0,0),5)
    
    #draw lines in bird's eye view of the perspective transform image
    cv2.line(warped,tuple(dst[0]),tuple(dst[1]),(255,0,0),5)
    cv2.line(warped,tuple(dst[1]),tuple(dst[2]),(255,0,0),5)
    cv2.line(warped,tuple(dst[2]),tuple(dst[3]),(255,0,0),5)
    cv2.line(warped,tuple(dst[3]),tuple(dst[0]),(255,0,0),5)

    showPlotBeforeAfter(undist,warped,"undistorted straight line 1","persperctive transform",axis='off')
    #showPlotBeforeAfter(undist,warped,"undistorted straight line 2","persperctive transform",axis='off')
    
    #Applying perspective transform on thresholded binary image (SobelX + S threshold)
    #combined
    image = mpimg.imread("test_images/test5.jpg")
    warped = cv2.warpPerspective(combined, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    showPlotBeforeAfter(combined,warped,"Combination of S and SobelX Gradient","binary warped",cmap='gray')
    
    #execute sliding windows to find lane lines and performs polyfit updating global variables left_fit and right_fit 
    sliding_window(combined,image)   

    #try pipe line in test image 
    image = mpimg.imread("test_images/test1.jpg")
    result = pipelines(image)
    plt.imshow(result)
    
    #run pipeline on "project_video.mp4" and save the 
    #annotated video to project5_AnnotatedVideo.mp4" 
    white_output = "project5_AnnotatedVideo.mp4"
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)






    