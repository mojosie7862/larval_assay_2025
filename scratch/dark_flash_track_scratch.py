import numpy as np
import cv2
# from TrackedObject import TrackedObject as TO
import matplotlib.pyplot as plt
import os
import re  # regular expressions library
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import remove_small_objects
import scipy.misc
import scipy.interpolate
import scipy.io
import pims
from tqdm import tqdm
import pandas as pd
import math
import sys

import scipy.stats as stats
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn import metrics

fn = '241217-nrxn1a_wd_1a_hetinx-'

# set directory containing image files
video_dir = "E:/jm_laCie_larval_behavior_avi_files/"
save_dir = "E:/jm_laCie_larval_behavior_tracked_data/"

# names of videos to track
Name = fn + 'DF_.avi'

# set gene name
Gene = "test"

# set directory containing background files
inDirbg = video_dir

# names of videos for background
Namebg1 = fn + 'DF_.avi'

# set the directory in which to save the files

SaveTrackingIm = True

exp_tracking_dir = save_dir + '/' + Name + '/'
exp_tracking_img_dir = save_dir + '/' + Name + '/trackedimages/'

# Number of fish to track
nFish = 100

# Number of rows and columns of wells
nRow = 10
nCol = 10

# Number of pixels in x and y direction
nPixel = 1024

# frames per tap event
nFrameseventtap = 500

# number of tap events
nTapevents = 42

# number of total blocks
nBlocks = 3

# frame rate in Hz
framerate = 500

# time between frames in milliseconds
framelength = 1 / framerate * 1000

# frame of tapstimulus; stimulus is after 30ms
tapstimframe = 0

# designate frame where angle about threshold would be deemed too early and removed
tooearly = 15

# absolute total curvature in radians before considered a candidate for slc or llc, or for react
obendthreshangle = 1.75
reactthreshangle = 0.5

# this is to say if there is no folder, can make one
os.makedirs(os.path.dirname(exp_tracking_dir), exist_ok=True)
os.makedirs(os.path.dirname(exp_tracking_img_dir), exist_ok=True)

# pims.open('C:/Users/phili_000/Documents/PENN/Granato Lab/Programming/test images+videos/CaSR22_09A_.avi')
video = pims.PyAVReaderIndexed(video_dir + Name)
# video2 = pims.Video('C:/Users/phili_000/Documents/PENN/Granato Lab/Programming/test images+videos/CaSR22_09A_.avi')

# nFrames = len(video)
nFrames = 21000

# pims.open('C:/Users/phili_000/Documents/PENN/Granato Lab/Programming/test images+videos/CaSR22_09A_.avi')
videobg1 = pims.PyAVReaderIndexed(video_dir + Namebg1)
# video2 = pims.Video('C:/Users/phili_000/Documents/PENN/Granato Lab/Programming/test images+videos/CaSR22_09A_.avi')

nFramesbg = len(videobg1)

# background subtraction

fr1 = videobg1[0][:, :, 0]
fr2 = videobg1[np.round(500)][:, :, 0]
fr3 = videobg1[np.round(1000)][:, :, 0]
fr4 = videobg1[np.round(1500)][:, :, 0]
fr5 = videobg1[np.round(2000)][:, :, 0]
fr6 = videobg1[np.round(2500)][:, :, 0]
fr7 = videobg1[np.round(3000)][:, :, 0]
fr8 = videobg1[np.round(3500)][:, :, 0]
fr9 = videobg1[np.round(4000)][:, :, 0]
fr10 = videobg1[np.round(4500)][:, :, 0]
fr11 = videobg1[np.round(5000)][:, :, 0]
fr12 = videobg1[np.round(5500)][:, :, 0]
fr13 = videobg1[np.round(6000)][:, :, 0]
fr14 = videobg1[np.round(6500)][:, :, 0]
fr15 = videobg1[np.round(7000)][:, :, 0]

# fr1 = videobg[0][:,:,0]
# fr2 = videobg[np.round(2000)][:,:,0]
# fr3 = videobg[np.round(2500)][:,:,0]
# fr4 = videobg[np.round(3000)][:,:,0]
# fr5 = videobg[np.round(3500)][:,:,0]
# fr6 = videobg[np.round(4000)][:,:,0]
# fr7 = videobg[np.round(4500)][:,:,0]
# fr8 = videobg[np.round(5000)][:,:,0]
# fr9 = videobg[np.round(5500)][:,:,0]
# fr10 = videobg[np.round(6000)][:,:,0]
# fr11 = videobg[np.round(6500)][:,:,0]

bg1 = np.median(np.stack((fr1, fr2, fr3, fr4, fr5, fr6, fr7, fr8, fr9, fr10, fr11, fr12, fr13, fr14, fr15), axis=2),
                axis=2)

bg1 = cv2.GaussianBlur(bg1.astype(np.float64), (3, 3), 0)

# bg1 = np.load(inDir + 'Results/' + Gene + '/bg1.npy')


# normalize pixels to 0 to 1
# divide = bg/256

# cv2.imshow('background', divide)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()


nSeg = 6
segLen1 = 4.75
segLen2 = 4.75
SearchAngle1 = np.pi / 2
SearchAngle2 = np.pi / 2.5
SearchAngle3 = np.pi / 2.25
tailThresh = 4


def tracking(bg, img, index):
    # Define some variables that are used to track the points along the back

    # Do a background subtraction, and blur the background
    mask = abs(bg - img)
    # mask[mask < 0] = 0

    # Do I need to do a gaussian blur?

    mask_smooth = cv2.GaussianBlur(mask, (3, 3), 0)

    # min_val = -50.0
    # img_crop = np.where(mask < min_val, img_crop, np.zeros_like(img_crop))

    # threshold the mask
    # th_img = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)[1]
    th_img = cv2.threshold(mask_smooth, 22, 255, cv2.THRESH_BINARY)[1]

    # label the connected components in the thresholded mask
    lb = label(th_img)

    # remove small objects from thresholded image
    lb_sorem = remove_small_objects(lb, 40)

    # re-label since small objects removes so get 1 to nFish
    lb2 = label(lb_sorem)

    # get statistics on these components - takes a labeled input image and returns
    # a list of RegionProperties - area, centroid among others
    props = regionprops(lb2)

    totobj = np.amax(lb2)
    print(str(totobj) + "in" + str(index + 1))
    # if np.max(props[:].label)!=nFish-1:
    # print("number of fish does not equal number of objects identified")
    # print("image number " + str(index))
    # quit()

    # Make an overlay image so we can save them to inspect where the tracked points
    # are getting placed on the image
    # overlay = img
    im_height, im_width = np.shape(mask_smooth)

    # copy the smoothed mask into a temporary image
    im_tmp = np.copy(mask_smooth)

    # make single dimensional arrays the size of nSeg to fill with x,y coordinates
    # of the tracked points. Fill their initial entries with nans
    # medInt = np.median(mask_smooth)

    y_cent = np.zeros(nFish)
    y_cent[:] = np.nan
    x_cent = np.zeros(nFish)
    x_cent[:] = np.nan

    pts_x = np.zeros((nFish, nSeg))
    pts_y = np.zeros((nFish, nSeg))
    pts_x[:, :] = np.nan
    pts_y[:, :] = np.nan

    angles = np.zeros((nFish, nSeg))
    angles[:, :] = np.nan

    for ind in range(0, totobj):
        # if totobj > nFish:
        #    break
        min_row, min_col, max_row, max_col = props[ind].bbox
        # if max(max_row - min_row, max_col - min_col) >30:
        #    continue

        # set curveaccum to allow to input total
        curveaccum = 0

        # find the centroid of
        cent = props[ind].centroid
        fishrow = 0
        fishcol = 0
        # go through and get row and column of this fish so can use it to index array; fish numbered from 1 in top left to 36 in bottorm right
        for row in range(1, nRow + 1):

            if cent[0] < row * nPixel / nRow:
                fishrow = row
                break

        for col in range(1, nCol + 1):
            if cent[1] < col * nPixel / nCol:
                fishcol = col
                break

        # this will return number of fish from 1 to nFish starting in top left corner with 1 and going L to R
        fishnum = ((fishrow - 1) * nCol) + fishcol

        # if there is already data in arrays for this fish number, go to top of for loop for next object, put 3 if statements so will remove if up to 4 objects
        if np.isfinite(y_cent[fishnum - 1]):
            y_cent[fishnum - 1] = np.nan
            x_cent[fishnum - 1] = np.nan
            curvature[fishnum - 1, index] = np.nan
            orients[fishnum - 1, index] = np.nan
            continue
        if np.isfinite(pts_y[fishnum - 1, 0]):
            continue
        if np.isfinite(pts_x[fishnum - 1, 0]):
            continue

        # find the centroid of the largest component
        y_cent[fishnum - 1], x_cent[fishnum - 1] = cent

        # find the bounding box of the largest component
        # min_row, min_col, max_row, max_col = props[ind].bbox

        # get the coordinate list for the pixels in this component
        coords = props[ind].coords

        # Using these coordinates, search this area of the mask_smooth in order to find the
        # point in the image with the largest diffence from the background. The coords
        # returns a list of (row,col)
        # In theory, this should be the head. Save these coordinates; np.argmax Returns
        # the indices of the maximum values along an axis.

        headInd = np.argmax(mask_smooth[coords[:, 0], coords[:, 1]])
        y_0 = coords[headInd, 0]
        x_0 = coords[headInd, 1]
        #        [y, x] = np.where(th_img)

        # not operator returns True if the statement is false; so if y_0 is Not A number (NaN), returns
        # True and then if does not occur. if NOT NaN, returns False and if occurs
        if not np.isnan(y_0):

            # Put the centroid and head points on the overlay;
            # .astype returns Copy of the array, cast to a specified type, here int
            # overlay[y_0.astype(int), x_0.astype(int)] = 255
            # overlay[y_cent[fishnum-1].astype(int), x_cent[fishnum-1].astype(int)] = 255

            # set first angle in array
            # angles[fishnum-1,0] = search_angle

            # Assign to the first coordinates in x,y coord arrays, the centroid coordinates
            #   pts_x[fishnum-1,0] = x_0;
            #   pts_y[fishnum-1,0] = y_0;
            pts_x[fishnum - 1, 0] = x_cent[fishnum - 1];
            pts_y[fishnum - 1, 0] = y_cent[fishnum - 1];

            # Not sure what this is for
            # curve = 0;

            # set the head point to 0 in this temporary image
            #   im_tmp[y_0.astype(int), x_0.astype(int)] = 0;
            im_tmp[y_cent[fishnum - 1].astype(int), x_cent[fishnum - 1].astype(int)] = 0;

            # loop over nSeg - 1 so we find (nSeg - 1) more points along the fish's back
            i = 0
            reverse = 0
            while i < nSeg - 1:
                # rint(str(fishnum-1) + '-' + str(index) + '-' + str(i))
                # update search direction for subsequent points
                if i > 1:
                    search_angle = np.arctan2(pts_y[fishnum - 1, i] - pts_y[fishnum - 1, i - 1],
                                              pts_x[fishnum - 1, i] - pts_x[fishnum - 1, i - 1])
                    ang = np.arange(search_angle - SearchAngle3, search_angle + SearchAngle3, np.pi / 20)
                    angles[fishnum - 1, i - 1] = search_angle
                    # sin(ang) * segLen will give y coordinate; cos(ang) * segLen will give x coord
                    # so, this line should give several y and x values for each angle
                    ArcCoors = pts_y[fishnum - 1, i] + np.sin(ang) * segLen2, pts_x[fishnum - 1, i] + np.cos(
                        ang) * segLen2

                if i == 1:
                    search_angle = np.arctan2(pts_y[fishnum - 1, i] - pts_y[fishnum - 1, i - 1],
                                              pts_x[fishnum - 1, i] - pts_x[fishnum - 1, i - 1])
                    ang = np.arange(search_angle - SearchAngle2, search_angle + SearchAngle2, np.pi / 20)
                    angles[fishnum - 1, i - 1] = search_angle
                    # sin(ang) * segLen will give y coordinate; cos(ang) * segLen will give x coord
                    # so, this line should give several y and x values for each angle
                    ArcCoors = pts_y[fishnum - 1, i] + np.sin(ang) * segLen2, pts_x[fishnum - 1, i] + np.cos(
                        ang) * segLen2

                # calculate search coordinates ; np.arange takes (START value, STOP value, increment)
                # Start value is included in the range, Stop value is not, will return an array
                # So, this will give an array with pi/20 spaced values btween this arc
                if i == 0:
                    if reverse == 0:
                        # define the angle, along which to search for the desired points
                        # arctan2 gives the angle in radians between teh positive x-axis and the ray to point x,y
                        # arctan2 changes based on sign of y, x and as such will return a value btween -pi and +pi
                        # for example, if x and y<0, computes arctany/x - pi
                        search_angle = np.arctan2(y_cent[fishnum - 1] - y_0, x_cent[fishnum - 1] - x_0)
                        # print("reverse=0, finshnum="+str(fishnum))

                    if reverse == 1:
                        search_angle = np.arctan2(y_cent[fishnum - 1] - pts_y[fishnum - 1, 1],
                                                  x_cent[fishnum - 1] - pts_x[fishnum - 1, 1])

                        # print("reverse=1, finshnum="+str(fishnum))

                    ang = np.arange(search_angle - SearchAngle1, search_angle + SearchAngle1, np.pi / 20)
                    # sin(ang) * segLen will give y coordinate; cos(ang) * segLen will give x coord
                    # so, this line should give several y and x values for each angle
                    ArcCoors = pts_y[fishnum - 1, i] + np.sin(ang) * segLen1, pts_x[fishnum - 1, i] + np.cos(
                        ang) * segLen1

                # np.asarray converts input to an array with dtype whatever is specified
                # so, arccoords should be an array of coordinates to search along
                ArcCoors = np.asarray(ArcCoors, dtype=int)

                # avoid out of bounds issues
                ArcCoors[0][ArcCoors[0] <= 0] = 1
                ArcCoors[0][ArcCoors[0] >= im_height] = im_height - 1
                ArcCoors[1][ArcCoors[1] <= 0] = 1
                ArcCoors[1][ArcCoors[1] >= im_width] = im_width - 1

                # ArcCoords[0] refers to all y positions, [1] refers to all x positions
                if reverse == 0:
                    im_tmp[ArcCoors[0], ArcCoors[1]] = 75
                if reverse == 1:
                    im_tmp[ArcCoors[0], ArcCoors[1]] = 125

                # find brightest pixel in smoothed image
                pt_max = np.max(mask_smooth[ArcCoors[0], ArcCoors[1]])

                # Returns the indices of the maximum value along an axis.
                pt_ind = np.argmax(mask_smooth[ArcCoors[0], ArcCoors[1]])
                pt_y = ArcCoors[0][pt_ind]
                pt_x = ArcCoors[1][pt_ind]

                # populate angles array with heading direction

                # check to see if first orientation was done in correct direction. Sometimes it mistakenly takes swimbladder
                # as the maximal point which sets off 180 degree difference from if had chosen the head
                # if i == 1:
                # if np.sign(angles[fishnum-1,i]) != np.sign(angles[fishnum-1,i-1]):
                # if angles[fishnum-1,i-1] < 0:
                # angles[fishnum-1,i-1] = angles[fishnum-1,i-1] + np.pi
                # if angles[fishnum-1,i-1] > 0:
                # angles[fishnum-1,i-1] = angles[fishnum-1,i-1] - np.pi

                # assign this as the center of the tail if the pixel is above a threshold of intesnity
                # (this is to avoid assigning points to space not actually occupied by the fish, like if it is short,
                # or we have gotten off track)
                if pt_max > tailThresh:
                    if reverse == 0:
                        im_tmp[pt_y.astype(int), pt_x.astype(int)] = 150
                    if reverse == 1:
                        im_tmp[pt_y.astype(int), pt_x.astype(int)] = 250

                    pts_y[fishnum - 1, i + 1] = pt_y
                    pts_x[fishnum - 1, i + 1] = pt_x

                    # add to curve accumlator
                    if i > 1:
                        # check if angle is between 90 and 180 because run into problems when it crosses x axis
                        if abs(angles[fishnum - 1, i - 2]) > np.pi / 2:
                            # if it is between 90 and 180, check to see if sign changes
                            if np.sign(angles[fishnum - 1, i - 2]) != np.sign(angles[fishnum - 1, i - 1]):
                                if angles[fishnum - 1, i - 1] > 0:
                                    curveaccum = curveaccum - ((np.pi - angles[fishnum - 1, i - 1]) + (
                                                np.pi + angles[fishnum - 1, i - 2]))
                                if angles[fishnum - 1, i - 1] < 0:
                                    curveaccum = curveaccum + ((np.pi + angles[fishnum - 1, i - 1]) + (
                                                np.pi - angles[fishnum - 1, i - 2]))
                            else:
                                curveaccum = curveaccum + (angles[fishnum - 1, i - 1] - angles[fishnum - 1, i - 2])
                        else:
                            curveaccum = curveaccum + (angles[fishnum - 1, i - 1] - angles[fishnum - 1, i - 2])

                    if i == nSeg - 2:
                        # if this is first frame of an event, should be equal to curveaccum as nothign to compare it to
                        if index % nFrameseventtap == 0:
                            if abs(curveaccum) > np.pi / 2:
                                if reverse == 1:
                                    curvature[fishnum - 1, index] = np.nan
                                    # print(str(fishnum) + " for " + str(curvature[fishnum-1]) + " in i " + str(i))
                                    orients[fishnum - 1, index] = np.nan
                                    break
                                if reverse == 0:
                                    # switch direction of initial orientation
                                    reverse += 1
                                    # reset curveaccumulator
                                    curveaccum = 0
                                    # reset counter
                                    i = 0
                                    continue
                            else:
                                curvature[fishnum - 1, index] = curveaccum
                                # print(str(fishnum) + " for " + str(curvature[fishnum-1]) + " in i " + str(i))
                                orients[fishnum - 1, index] = angles[fishnum - 1, 0]

                        # if not first frame and if too big of a jump between frames, reverse initial direction unless this has already been done
                        else:
                            # if curveaccum is >0.8 and previous is NaN, make this one NaN as well as will not be able to tell
                            # check if difference is greater than pi, if so, subtract from 2pi to get true difference
                            if np.isnan(orients[fishnum - 1, index - 1]):
                                curvature[fishnum - 1, index] = np.nan
                                # print(str(fishnum) + " for " + str(curvature[fishnum-1]) + " in i " + str(i))
                                orients[fishnum - 1, index] = np.nan
                                break

                            orients[fishnum - 1, index] = angles[fishnum - 1, 0]

                            if abs(orients[fishnum - 1, index - 1] - orients[fishnum - 1, index]) > np.pi:
                                deltaorient = (2 * np.pi) - abs(
                                    orients[fishnum - 1, index - 1] - orients[fishnum - 1, index])
                            else:
                                deltaorient = abs(orients[fishnum - 1, index - 1] - orients[fishnum - 1, index])

                            if deltaorient > np.pi / 2:
                                print(str(fishnum) + " for " + str(deltaorient) + " in index " + str(index))
                                if reverse == 1:
                                    curvature[fishnum - 1, index] = np.nan
                                    # print(str(fishnum) + " for " + str(curvature[fishnum-1]) + " in i " + str(i))
                                    orients[fishnum - 1, index] = np.nan
                                    break
                                if reverse == 0:
                                    # switch direction of initial orientation
                                    reverse += 1
                                    # reset curveaccumulator
                                    curveaccum = 0
                                    # reset counter
                                    i = 0
                                    continue
                            else:
                                curvature[fishnum - 1, index] = curveaccum
                                # print(str(fishnum) + " for " + str(curvature[fishnum-1]) + " in i " + str(i))
                                orients[fishnum - 1, index] = angles[fishnum - 1, 0]

                    i += 1

                else:  # if we didnt find enough point on the tail, try to reverse initial search unless this has already been done
                    if i <= (nSeg - 2):

                        if index % nFrameseventtap == 0:

                            if reverse == 1:
                                curvature[fishnum - 1, index] = np.nan
                                # print(str(fishnum) + " for " + str(curvature[fishnum-1]) + " in i " + str(i))
                                orients[fishnum - 1, index] = np.nan
                                break
                            if reverse == 0:
                                # switch direction of initial orientation
                                reverse += 1
                                # reset curveaccumulator
                                curveaccum = 0
                                # reset counter
                                i = 0
                                continue
                        else:
                            if np.isnan(orients[fishnum - 1, index - 1]):
                                curvature[fishnum - 1, index] = np.nan
                                # print(str(fishnum) + " for " + str(curvature[fishnum-1]) + " in i " + str(i))
                                orients[fishnum - 1, index] = np.nan
                                break

                            orients[fishnum - 1, index] = angles[fishnum - 1, 0]

                            if abs(orients[fishnum - 1, index - 1] - orients[fishnum - 1, index]) > np.pi:
                                deltaorient = (2 * np.pi) - abs(
                                    orients[fishnum - 1, index - 1] - orients[fishnum - 1, index])
                            else:
                                deltaorient = abs(orients[fishnum - 1, index - 1] - orients[fishnum - 1, index])

                            if deltaorient > np.pi / 2:
                                print(str(fishnum) + " for " + str(deltaorient) + " in index " + str(index))
                                if reverse == 1:
                                    curvature[fishnum - 1, index] = np.nan
                                    # print(str(fishnum) + " for " + str(curvature[fishnum-1]) + " in i " + str(i))
                                    orients[fishnum - 1, index] = np.nan
                                    break
                                if reverse == 0:
                                    # switch direction of initial orientation
                                    reverse += 1
                                    # reset curveaccumulator
                                    curveaccum = 0
                                    # reset counter
                                    i = 0
                                    continue
                            else:
                                curvature[fishnum - 1, index] = np.nan
                                # print(str(fishnum) + " for " + str(curvature[fishnum-1]) + " in i " + str(i))
                                orients[fishnum - 1, index] = angles[fishnum - 1, 0]
                                break

                    else:
                        curvature[fishnum - 1, index] = np.nan
                        # print(str(fishnum) + " for " + str(curvature[fishnum-1]) + " in i " + str(i))
                        orients[fishnum - 1, index] = angles[fishnum - 1, 0]
                        break

    return im_tmp, pts_y, pts_x, y_cent, x_cent, curvature, orients


# Initialize an array to hold the coordinates of the 8 tracked points (x and y coords)
# for each tracked image. We want the array to be size 8, 2, page_num, where nFrames
# is the number of images we are tracking. We will assign y coordinates as the first column
# index i.e. top_coords[:,0,:] and x coordinates as the second index i.e. top_coords[:,1,:]
# This is to be consistent with all of the other code used for this project
xycoords = np.zeros((nFish, nSeg, 2, nFrames))
cents = np.zeros((nFish, 2, nFrames))
curvature = np.zeros((nFish, nFrames))
orients = np.zeros((nFish, nFrames))
orients[:] = np.nan

curvature[:] = np.nan

# tracking

print("Tracking....")

# tqdm shows a progress bar around any iterable
for index, img in enumerate(tqdm(video)):

    if (index < nFrames):
        # .copy() will maintain the origical list unchanged so that if you change the new list it
        # won't affect old one
        # img = plt.imread(inDir + "\\" + filename).copy()

        img = img[:, :,
              0]  # its actually a black and white image, but gets read in as three channel. This is probably inefficient at some point

        # do I need to do gaussian blur?
        img = cv2.GaussianBlur(img.astype(np.float64), (3, 3), 0)

        # img = img[0:800,0:1075]

        # calling the tracking image with bg and individual img
        timg, pts_y, pts_x, y_cent, x_cent, curvature, orients = tracking(bg1, img, index)

        cents[:, 0, index] = y_cent
        cents[:, 1, index] = x_cent
        xycoords[:, :, 0, index] = pts_y
        xycoords[:, :, 1, index] = pts_x

        # save the tracked coordinates and the tracked image files
        # scipy.io.savemat(dest)
        if SaveTrackingIm:
            # scipy.misc.imsave - (name of output file, array containing image values,format=image format unless
            # specified in name of file)
            # np.vstack - Stack arrays in sequence vertically (row wise)
            # timg here is the im_tmp object returned from tracking loop
            cv2.imwrite(exp_tracking_img_dir + Name + '_' + str(index) + '.png', np.vstack((timg)))

# np. savez - Save several arrays into a single file in uncompressed .npz format.
# (file name, designed array names)
print("Saving tracking data arrays....")
np.savez(exp_tracking_dir + '/TrackData.npz', xycoords=xycoords, cents=cents, curvature=curvature, orients=orients)

import seaborn as sns
from scipy.signal import medfilt

# setup med-filtered arrays
medfiltcentY = np.zeros((nFish, nFrames))
medfiltcentY[:, :] = np.nan
medfiltcentX = np.zeros((nFish, nFrames))
medfiltcentX[:, :] = np.nan

for zf_num in range(nFish):
    for i in range(0, nTapevents):
        medfiltcentY[zf_num, (i * nFrameseventtap):((i + 1) * nFrameseventtap)] = medfilt(
            cents[zf_num, 0, (i * nFrameseventtap):((i + 1) * nFrameseventtap)], 3)
        medfiltcentX[zf_num, (i * nFrameseventtap):((i + 1) * nFrameseventtap)] = medfilt(
            cents[zf_num, 1, (i * nFrameseventtap):((i + 1) * nFrameseventtap)], 3)

    # np.sqrt() is square root; np.square will give you ''2 of element-wise array
    # np.diff calculates the nth discrete difference along given axis - so, gives difference between values and returns
    # and array of differences
    # speed = sqrt(displacement in y squared + displacement in x squared) - So why is there a comma instead of  +??

    # pixframe defined in milliseconds (difference in time for each frame)
pixframe = 1
# speed = np.sqrt(np.square(np.diff(headY)), np.square(np.diff(headX)))


# START CALCULATIONS HERE

# number of event blocks
nBlockshab = 3

disp = np.zeros((nFish, nFrames - 1))
disp[:, :] = np.nan
filtdisp = np.zeros((nFish, nFrames - 1))
filtdisp[:, :] = np.nan

angdisp = np.zeros((nFish, nFrames - 1))
angdisp[:, :] = np.nan

angdispmax = np.zeros((nFish, nTapevents))
angdispmax[:, :] = np.nan

latencytaps = np.zeros((nFish, nTapevents))
latencytaps[:, :] = np.nan
latencytapsms = np.zeros((nFish, nTapevents))
latencytapsms[:, :] = np.nan
slctaps = np.zeros((nFish, nTapevents))
slctaps[:, :] = np.nan
llctaps = np.zeros((nFish, nTapevents))
llctaps[:, :] = np.nan
reacttaps = np.zeros((nFish, nTapevents))
reacttaps[:, :] = np.nan
nomvmttaps = np.zeros((nFish, nTapevents))
nomvmttaps[:, :] = np.nan
tapbendmax = np.zeros((nFish, nTapevents))
tapbendmax[:, :] = np.nan
tapdeltaorient = np.zeros((nFish, nTapevents))
tapdeltaorient[:, :] = np.nan
tapdisplacement = np.zeros((nFish, nTapevents))
tapdisplacement[:, :] = np.nan
tapdistance = np.zeros((nFish, nTapevents))
tapdistance[:, :] = np.nan
movementyesno = np.zeros((nFish, nFrames - 1))
movementyesno[:, :] = np.nan
movementyesnofill = np.zeros((nFish, nFrames - 1))
movementyesnofill[:, :] = np.nan
tapduration = np.zeros((nFish, nTapevents))
tapduration[:, :] = np.nan

tapright = np.zeros((nFish, nTapevents))
tapright[:, :] = np.nan
taptowards = np.zeros((nFish, nTapevents))
taptowards[:, :] = np.nan
taptowardshalf = np.zeros((nFish, nTapevents))
taptowardshalf[:, :] = np.nan

maxangvelocity = np.zeros((nFish, nTapevents))
maxangvelocity[:, :] = np.nan

print("Calculating displacement....")
# calculating displacement and thigmotaxis
for zf_num in range(nFish):
    i = 0
    while i < nFrames:
        if np.isnan(cents[zf_num, 0, i:((math.floor(i / nFrameseventtap) + 1) * nFrameseventtap)]).any():
            # if no centroid is defined in any frame of this event, skip to next block
            event = math.floor(i / nFrameseventtap)
            i = (event + 1) * nFrameseventtap
            continue

        else:

            # calculate displacement between each frame
            if i > 0:
                disp[zf_num, i - 1] = np.sqrt((cents[zf_num, 0, i] - cents[zf_num, 0, i - 1]) ** 2 + (
                            cents[zf_num, 1, i] - cents[zf_num, 1, i - 1]) ** 2)
                filtdisp[zf_num, i - 1] = np.sqrt((medfiltcentY[zf_num, i] - medfiltcentY[zf_num, i - 1]) ** 2 + (
                            medfiltcentX[zf_num, i] - medfiltcentX[zf_num, i - 1]) ** 2)
                # define movement as displacement >0.2 and populate array with 1 for >0.2 and 0 for <0.2; nan if displ nan
                if filtdisp[zf_num, i - 1] > 0.5:
                    if i % nFrameseventtap == 0:
                        movementyesno[zf_num, i - 1] = 0
                    else:
                        movementyesno[zf_num, i - 1] = 1
                else:
                    movementyesno[zf_num, i - 1] = 0

            i = i + 1

print("Calculating angular displacement....")
# calculating displacement and thigmotaxis
for zf_num in range(nFish):
    i = 0
    while i < nFrames:
        if np.isnan(orients[zf_num, i:((math.floor(i / nFrameseventtap) + 1) * nFrameseventtap)]).any():
            # if no orient is defined in any frame of this event, skip to next block
            event = math.floor(i / nFrameseventtap)
            i = (event + 1) * nFrameseventtap
            continue

        else:
            # calculate angular displacement between each frame
            if i > 0:

                # check if difference is greater than pi, if so, subtract from 2pi to get true difference
                if abs(orients[zf_num, i] - orients[zf_num, i - 1]) > np.pi:
                    angdisp[zf_num, i - 1] = (2 * np.pi) - abs(orients[zf_num, i] - orients[zf_num, i - 1])
                else:
                    angdisp[zf_num, i - 1] = abs(orients[zf_num, i] - orients[zf_num, i - 1])
            i = i + 1

print("Defining movement....")
# Attempt to fill holes of movement array. Will eliminate singleton 1s or 0s so that for movement will need >1 frame with value of
# 1 and to stop movement will need >1 frame with value of 0
# fill initial frame with values of initial array
movementyesnofill[:, 0] = movementyesno[:, 0]

for zf_num in range(nFish):
    for i in range(1, nFrames - 1):
        if np.isnan(movementyesno[zf_num, i]):
            continue
        # if nan either before or after value i, just keep value because cannot tell
        if np.isnan(movementyesnofill[zf_num, i - 1]):
            movementyesnofill[zf_num, i] = movementyesno[zf_num, i]
            continue
        if i < nFrames - 2:
            if np.isnan(movementyesno[zf_num, i + 1]):
                movementyesnofill[zf_num, i] = movementyesno[zf_num, i]
                continue

        # here is where we define movement and stopping as >1 frame of either 1 or 0
        if movementyesno[zf_num, i] == 0:
            if movementyesnofill[zf_num, i - 1] == 0:
                movementyesnofill[zf_num, i] = 0
            if movementyesnofill[zf_num, i - 1] == 1:
                if i < nFrames - 2:
                    if movementyesno[zf_num, i + 1] == 1:
                        movementyesnofill[zf_num, i] = 1
                    if movementyesno[zf_num, i + 1] == 0:
                        movementyesnofill[zf_num, i] = 0
                else:
                    movementyesnofill[zf_num, i] = 0

        if movementyesno[zf_num, i] == 1:
            if movementyesnofill[zf_num, i - 1] == 1:
                movementyesnofill[zf_num, i] = 1
            if movementyesnofill[zf_num, i - 1] == 0:
                if i < nFrames - 2:
                    if movementyesno[zf_num, i + 1] == 0:
                        movementyesnofill[zf_num, i] = 0
                    if movementyesno[zf_num, i + 1] == 1:
                        movementyesnofill[zf_num, i] = 1
                else:
                    movementyesnofill[zf_num, i] = 1

print("Finding tap latency and classifying movement....")
# finding tap response frame and latency
for zf_num in range(nFish):
    i = 0
    while i < nFrames:
        # calculating latency to bend

        # if this frame and next frame have curvatures of NaN, eliminate this event - make latency Nan, then skip to next event
        if np.isnan(curvature[zf_num, i]) and np.isnan(curvature[zf_num, i + 1]):
            event = math.floor(i / nFrameseventtap)
            latencytaps[zf_num, event] = np.nan
            i = (event + 1) * nFrameseventtap
            continue
        # if only this frame has curvature of Nan, go to next frame
        if np.isnan(curvature[zf_num, i]):
            i = i + 1
            continue
        else:
            event = math.floor(i / nFrameseventtap)

            if np.nanmax(abs(curvature[zf_num, ((event * nFrameseventtap) + tapstimframe):(
                    (event + 1) * nFrameseventtap)])) > obendthreshangle:
                if abs(curvature[zf_num, i]) > 1:
                    latencytaps[zf_num, event] = i - (nFrameseventtap * event)
                    if latencytaps[zf_num, event] <= tooearly:
                        latencytaps[zf_num, event] = np.nan
                    else:
                        # classify as obend
                        slctaps[zf_num, event] = 1
                    i = (event + 1) * nFrameseventtap
                    continue
                else:
                    if (i + 1) % nFrameseventtap == 0:
                        nomvmttaps[zf_num, event] = 1
                        latencytaps[zf_num, event] = np.nan
                    i = i + 1
                    continue
            if np.nanmax(abs(curvature[zf_num, ((event * nFrameseventtap) + tapstimframe):(
                    (event + 1) * nFrameseventtap)])) > reactthreshangle:
                if abs(curvature[zf_num, i]) > reactthreshangle:
                    latencytaps[zf_num, event] = i - (nFrameseventtap * event)
                    if latencytaps[zf_num, event] <= tooearly:
                        latencytaps[zf_num, event] = np.nan
                    else:
                        # classify as react
                        reacttaps[zf_num, event] = 1
                    i = (event + 1) * nFrameseventtap
                    continue
                else:
                    if (i + 1) % nFrameseventtap == 0:
                        nomvmttaps[zf_num, event] = 1
                        latencytaps[zf_num, event] = np.nan
                    i = i + 1
                    continue
            else:
                # classify as no movement
                nomvmttaps[zf_num, event] = 1
                latencytaps[zf_num, event] = np.nan
                i = (event + 1) * nFrameseventtap

# calculate latency in ms from frame zf_num where > certain radian
latencytapsms[:, :] = (latencytaps[:, :] * framelength) - (tapstimframe * framelength)

print("Calculating bend max, delta orient, total displacement....")
# calculating bend max, change in orientation, and totaldisplacement for taps
for zf_num in range(nFish):
    i = 0
    while i < nTapevents:
        # check if latency is before stimulus (ie. nan)
        if np.isnan(latencytaps[zf_num, i]) or latencytaps[zf_num, event] == nFrameseventtap - 1:
            i = i + 1
            continue

        else:
            # time moving
            tapduration[zf_num, i] = np.sum(movementyesnofill[zf_num, ((i * nFrameseventtap) + tapstimframe):(
                        ((i + 1) * nFrameseventtap) - 1)]) * framelength

            # label bouts in block
            lb = label(
                movementyesnofill[zf_num, ((i * nFrameseventtap) + tapstimframe):(((i + 1) * nFrameseventtap) - 1)])

            # remove bouts less than 2 - though shouldn't be any anyways
            lb_sorem = remove_small_objects(lb, 2)

            # re-label since small objects removed
            lb2 = label(lb_sorem)

            # define array of distances in this block
            x = filtdisp[zf_num, ((i * nFrameseventtap) + tapstimframe):(((i + 1) * nFrameseventtap) - 1)]

            # total distance: is equal to sum of displacement following tap stimulus where movementyesnofill is 1
            tapdistance[zf_num, i] = np.sum(x[lb2 != 0])

            # change in orientation
            # calculate frame right before tap
            tapframe = i * nFrameseventtap

            # check if difference is greater than pi, if so, subtract from 2pi to get true difference
            if abs(orients[zf_num, (nFrameseventtap * (i + 1)) - 1] - orients[zf_num, tapframe]) > np.pi:
                tapdeltaorient[zf_num, i] = (2 * np.pi) - abs(
                    orients[zf_num, (nFrameseventtap * (i + 1)) - 1] - orients[zf_num, tapframe])
            else:
                tapdeltaorient[zf_num, i] = abs(
                    orients[zf_num, (nFrameseventtap * (i + 1)) - 1] - orients[zf_num, tapframe])

            # tap displacement
            tapdisplacement[zf_num, i] = np.sqrt(
                (cents[zf_num, 0, tapframe] - cents[zf_num, 0, ((i + 1) * nFrameseventtap) - 1]) ** 2 + (
                            cents[zf_num, 1, tapframe] - cents[zf_num, 1, ((i + 1) * nFrameseventtap) - 1]) ** 2)

            # for react or nomovement cases, just check for max bend angle after tap
            if latencytaps[zf_num, i] == (nFrameseventtap - 1):
                i = i + 1
            else:
                # for other bend max: calculate frame where above threshold movement happens
                frameorient = i * nFrameseventtap + (latencytaps[zf_num, i]) - 20
                framecurve = i * nFrameseventtap + (latencytaps[zf_num, i])

                # compute L/R and towards/away; if orientation of fish is between 0 and pi
                if orients[zf_num, int(frameorient)] > 0:
                    # if curvature is negative, meaning CW
                    if curvature[zf_num, int(framecurve)] < 0:
                        tapright[zf_num, i] = 1
                        taptowards[zf_num, i] = 0

                    # if curvature is positive, meaning CCW
                    if curvature[zf_num, int(framecurve)] > 0:
                        tapright[zf_num, i] = 0
                        taptowards[zf_num, i] = 1
                    # if initial orient is within middle 90deg
                    if abs(orients[zf_num, int(frameorient)]) > (4.5 * np.pi / 18) and abs(
                            orients[zf_num, int(frameorient)]) < (13.5 * np.pi / 18):
                        if curvature[zf_num, int(framecurve)] < 0:
                            taptowardshalf[zf_num, i] = 0

                        if curvature[zf_num, int(framecurve)] > 0:
                            taptowardshalf[zf_num, i] = 1

                            # if orientation of fish is between 0 and pi
                if orients[zf_num, int(frameorient)] < 0:
                    # if curvature is negative, meaning CW
                    if curvature[zf_num, int(framecurve)] < 0:
                        tapright[zf_num, i] = 1
                        taptowards[zf_num, i] = 1

                    # if curvature is positive, meaning CCW
                    if curvature[zf_num, int(framecurve)] > 0:
                        tapright[zf_num, i] = 0
                        taptowards[zf_num, i] = 0

                    # if initial orient is within middle 90deg
                    if abs(orients[zf_num, int(frameorient)]) > (4.5 * np.pi / 18) and abs(
                            orients[zf_num, int(frameorient)]) < (13.5 * np.pi / 18):
                        if curvature[zf_num, int(framecurve)] < 0:
                            taptowardshalf[zf_num, i] = 1

                        if curvature[zf_num, int(framecurve)] > 0:
                            taptowardshalf[zf_num, i] = 0

                frame = (i * nFrameseventtap) + np.nanargmax(
                    abs(curvature[zf_num, (i * nFrameseventtap):((i + 1) * nFrameseventtap)]))
                j = i
                if frame + 2 > nFrames - 1:
                    i = i + 1
                    continue
                else:
                    while frame < (j + 1) * nFrameseventtap:
                        # check if curvature is nan around bend max, if so break while loop as will not be able to decipher bendmax
                        if np.isnan(curvature[zf_num, int(frame)]) or np.isnan(
                                curvature[zf_num, int(frame - 1)]) or np.isnan(
                                curvature[zf_num, int(frame - 2)]) or np.isnan(
                                curvature[zf_num, int(frame + 1)]) or np.isnan(curvature[zf_num, int(frame + 2)]):
                            j = j + 1
                            break
                        else:
                            # find point of local max right after threshold reached
                            if abs(curvature[zf_num, int(frame)]) > abs(curvature[zf_num, int(frame - 1)]) and abs(
                                    curvature[zf_num, int(frame)]) > abs(curvature[zf_num, int(frame + 1)]) and abs(
                                    curvature[zf_num, int(frame)]) > abs(curvature[zf_num, int(frame + 2)]):
                                tapbendmax[zf_num, j] = curvature[zf_num, int(frame)]
                                j = j + 1
                                break
                            # if at end of tap window, move on to next event
                            else:
                                frame = frame + 1
                                if frame == (j + 1) * nFrameseventtap:
                                    j = j + 1

                frame = i * nFrameseventtap + (latencytaps[zf_num, i])

                while frame < (i + 1) * nFrameseventtap:
                    print(str(zf_num))
                    # check if curvature is nan around bend max, if so break while loop as will not be able to decipher bendmax
                    if np.isnan(angdisp[zf_num, int(frame)]) or np.isnan(angdisp[zf_num, int(frame - 1)]) or np.isnan(
                            angdisp[zf_num, int(frame - 2)]) or np.isnan(angdisp[zf_num, int(frame - 3)]):
                        i = i + 1
                        break
                    else:
                        # find point of local max right after threshold reached
                        if abs(angdisp[zf_num, int(frame - 2)]) > abs(angdisp[zf_num, int(frame - 3)]) and abs(
                                angdisp[zf_num, int(frame - 2)]) > abs(angdisp[zf_num, int(frame - 1)]) and abs(
                                angdisp[zf_num, int(frame - 2)]) > abs(angdisp[zf_num, int(frame)]):
                            angdispmax[zf_num, i] = angdisp[zf_num, int(frame - 2)]
                            i = i + 1
                            break
                        # if at end of tap window, move on to next event
                        else:
                            frame = frame + 1
                            if frame + 3 > nFrames - 1:
                                i = i + 1
                                break
                            if frame == ((i + 1) * nFrameseventtap):
                                i = i + 1
                                break

print("Setting up final arrays....")
# deisgnate which fish are mut/het/wt
muts = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
     32, 33])
hets = np.array(
    [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
     63, 64, 65, 66])
wts = np.array(
    [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
     96, 97, 98, 99, 100])

slcnan = np.zeros((nFish, nBlocks))
slcnan[:] = np.nan
slctot = np.zeros((nFish, nBlocks))
slctot[:] = np.nan
latnan = np.zeros((nFish, nBlocks))
latnan[:] = np.nan
lattot = np.zeros((nFish, nBlocks))
lattot[:] = np.nan
tapbendmaxnan = np.zeros((nFish, nBlocks))
tapbendmaxnan[:] = np.nan
tapbendmaxtot = np.zeros((nFish, nBlocks))
tapbendmaxtot[:] = np.nan
tapdeltaorientnan = np.zeros((nFish, nBlocks))
tapdeltaorientnan[:] = np.nan
tapdeltaorienttot = np.zeros((nFish, nBlocks))
tapdeltaorienttot[:] = np.nan
tapdisplacementnan = np.zeros((nFish, nBlocks))
tapdisplacementnan[:] = np.nan
tapdisplacementtot = np.zeros((nFish, nBlocks))
tapdisplacementtot[:] = np.nan
tapdistancenan = np.zeros((nFish, nBlocks))
tapdistancenan[:] = np.nan
tapdistancetot = np.zeros((nFish, nBlocks))
tapdistancetot[:] = np.nan
angdispmaxnan = np.zeros((nFish, nBlocks))
angdispmaxnan[:] = np.nan
angdispmaxtot = np.zeros((nFish, nBlocks))
angdispmaxtot[:] = np.nan

avglat = np.zeros((nFish, nBlocks))
avglat[:] = np.nan
medlat = np.zeros((nFish, nBlocks))
medlat[:] = np.nan
slc = np.zeros((nFish, nBlocks))
slc[:] = np.nan
llc = np.zeros((nFish, nBlocks))
llc[:] = np.nan
react = np.zeros((nFish, nBlocks))
react[:] = np.nan
nomvmt = np.zeros((nFish, nBlocks))
nomvmt[:] = np.nan
avgbend = np.zeros((nFish, nBlocks))
avgbend[:] = np.nan
medbend = np.zeros((nFish, nBlocks))
medbend[:] = np.nan
avgorient = np.zeros((nFish, nBlocks))
avgorient[:] = np.nan
medorient = np.zeros((nFish, nBlocks))
medorient[:] = np.nan
avgdisp = np.zeros((nFish, nBlocks))
avgdisp[:] = np.nan
meddisp = np.zeros((nFish, nBlocks))
meddisp[:] = np.nan
avgdist = np.zeros((nFish, nBlocks))
avgdist[:] = np.nan
meddist = np.zeros((nFish, nBlocks))
meddist[:] = np.nan
avgduration = np.zeros((nFish, nBlocks))
avgduration[:] = np.nan
medduration = np.zeros((nFish, nBlocks))
medduration[:] = np.nan
avgangvelmax = np.zeros((nFish, nBlocks))
avgangvelmax[:] = np.nan
medangvelmax = np.zeros((nFish, nBlocks))
medangvelmax[:] = np.nan

right = np.zeros((nFish, nBlocks))
right[:] = np.nan
towards = np.zeros((nFish, nBlocks))
towards[:] = np.nan
towardshalf = np.zeros((nFish, nBlocks))
towardshalf[:] = np.nan

print("Tabulating results....")
# count number of nan and total events and make calculations for sensitivity events
for zf_num in range(nFish):
    for i in range(nBlockshab):
        slcnan[zf_num, i] = np.count_nonzero(np.isnan(latencytaps[zf_num, int(i * (nTapevents / nBlocks)):int(
            (i + 1) * (nTapevents / nBlocks))])) - np.count_nonzero(
            np.isfinite(nomvmttaps[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))
        slctot[zf_num, i] = np.size(
            latencytaps[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))])
        latnan[zf_num, i] = np.count_nonzero(
            np.isnan(latencytaps[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))
        lattot[zf_num, i] = np.size(
            latencytaps[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))])
        tapbendmaxnan[zf_num, i] = np.count_nonzero(
            np.isnan(tapbendmax[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))
        tapbendmaxtot[zf_num, i] = np.size(
            tapbendmax[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))])
        tapdeltaorientnan[zf_num, i] = np.count_nonzero(
            np.isnan(tapdeltaorient[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))
        tapdeltaorienttot[zf_num, i] = np.size(
            tapdeltaorient[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))])
        tapdisplacementnan[zf_num, i] = np.count_nonzero(
            np.isnan(tapdisplacement[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))
        tapdisplacementtot[zf_num, i] = np.size(
            tapdisplacement[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))])
        tapdistancenan[zf_num, i] = np.count_nonzero(
            np.isnan(tapdistance[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))
        tapdistancetot[zf_num, i] = np.size(
            tapdistance[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))])
        angdispmaxnan[zf_num, i] = np.count_nonzero(
            np.isnan(angdispmax[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))
        angdispmaxtot[zf_num, i] = np.size(
            angdispmax[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))])

        avglat[zf_num, i] = np.nanmean(
            latencytapsms[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))])
        medlat[zf_num, i] = np.nanmedian(
            latencytapsms[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))])
        slc[zf_num, i] = np.nansum(
            slctaps[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]) / (
                                     slctot[zf_num, i] - slcnan[zf_num, i])
        react[zf_num, i] = np.nansum(
            reacttaps[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]) / (
                                       slctot[zf_num, i] - slcnan[zf_num, i])
        nomvmt[zf_num, i] = np.nansum(
            nomvmttaps[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]) / (
                                        slctot[zf_num, i] - slcnan[zf_num, i])

        right[zf_num, i] = np.nansum(
            tapright[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]) / (
                                       lattot[zf_num, i] - latnan[zf_num, i])
        towards[zf_num, i] = np.nansum(
            taptowards[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]) / (
                                         lattot[zf_num, i] - latnan[zf_num, i])
        towardshalf[zf_num, i] = np.nansum(taptowardshalf[zf_num, int(i * (nTapevents / nBlocks)):int(
            (i + 1) * (nTapevents / nBlocks))]) / np.count_nonzero(
            np.isfinite(taptowardshalf[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))

        avgbend[zf_num, i] = np.nanmean(
            abs(tapbendmax[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))
        medbend[zf_num, i] = np.nanmedian(
            abs(tapbendmax[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))
        avgorient[zf_num, i] = np.nanmean(
            abs(tapdeltaorient[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))
        medorient[zf_num, i] = np.nanmedian(
            abs(tapdeltaorient[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))
        avgdisp[zf_num, i] = np.nanmean(
            abs(tapdisplacement[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))
        meddisp[zf_num, i] = np.nanmedian(
            abs(tapdisplacement[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))
        avgdist[zf_num, i] = np.nanmean(
            abs(tapdistance[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))
        meddist[zf_num, i] = np.nanmedian(
            abs(tapdistance[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))
        avgduration[zf_num, i] = np.nanmean(
            abs(tapduration[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))
        medduration[zf_num, i] = np.nanmedian(
            abs(tapduration[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))
        avgangvelmax[zf_num, i] = np.nanmean(
            abs(angdispmax[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))
        medangvelmax[zf_num, i] = np.nanmedian(
            abs(angdispmax[zf_num, int(i * (nTapevents / nBlocks)):int((i + 1) * (nTapevents / nBlocks))]))

print("Saving tracking data arrays....")
np.savez(exp_tracking_dir + '/AnalyzedData.npz', medfiltcentY=medfiltcentY, medfiltcentX=medfiltcentX, disp=disp,
         filtdisp=filtdisp,
         latencytaps=latencytaps, latencytapsms=latencytapsms, slctaps=slctaps,
         llctaps=llctaps, reacttaps=reacttaps, nomvmttaps=nomvmttaps, tapbendmax=tapbendmax,
         tapdeltaorient=tapdeltaorient, tapdisplacement=tapdisplacement, tapdistance=tapdistance,
         movementyesno=movementyesno,
         movementyesnofill=movementyesnofill,
         slcnan=slcnan, slctot=slctot, latnan=latnan, lattot=lattot, tapbendmaxnan=tapbendmaxnan,
         tapbendmaxtot=tapbendmaxtot,
         angdispmaxnan=angdispmaxnan, angdispmaxtot=angdispmaxtot, avgangvelmax=avgangvelmax, medangvelmax=medangvelmax,
         tapdeltaorientnan=tapdeltaorientnan, tapdeltaorienttot=tapdeltaorienttot,
         tapdisplacementnan=tapdisplacementnan, tapdisplacementtot=tapdisplacementtot,
         tapdistancenan=tapdistancenan, tapdistancetot=tapdistancetot, right=right, towards=towards,
         towardshalf=towardshalf,
         avglat=avglat, medlat=medlat, slc=slc, llc=llc, react=react, nomvmt=nomvmt, avgbend=avgbend, medbend=medbend,
         avgorient=avgorient, medorient=medorient, avgdisp=avgdisp, meddisp=meddisp, avgdist=avgdist, meddist=meddist,
         avgduration=avgduration, medduration=medduration
         )
