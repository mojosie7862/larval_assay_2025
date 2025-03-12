import numpy as np
import cv2
from skimage.draw import circle_perimeter
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import math
import os
import torch
from scipy.spatial.distance import euclidean
from sympy.geometry import Point, Triangle


def xy_center(fish_num):
    nCol = 10
    nRow = 10
    nPixel = 1024

    row = math.ceil((fish_num + 1) / nCol)
    col = (fish_num + 1) - (nCol * (row - 1))
    ycenter = (nPixel / nRow) / 2 + ((row - 1) * (nPixel / nRow))
    xcenter = (nPixel / nCol) / 2 + ((col - 1) * (nPixel / nCol))

    return xcenter, ycenter


def get_segment_coords(h, k, r, radian1, radian2, interval):
    '''
    :param h: x center coordinate
    :param k: y center coordinate
    :param r: radius
    :param radian1: start radian of segment
    :param radian2: end radian of segment
    :param interval: step determining # of lines/x,y coordinates output
    :return: list of x,y coordinates on the circle pertaining to specified segment
    '''
    coord_arr = []
    for degree in np.arange(radian1, radian2, interval):
        x = int(h + r * math.cos(degree))
        y = int(k + r * math.sin(degree))
        coord_arr.append([x, y])
    return coord_arr


def get_circle_x_y_eigths(h, k, r, radian_ls):
    '''
    :param h: x center coordinate
    :param k: y center coordinate
    :param r: radius
    :param radian_ls: list of radians at which to generate list of coordinates on the circle (should be length 8)
    :return: list of x,y coordinates on the circle corresponding to "social" regions of the well
    '''
    social_segment_1 = get_segment_coords(h, k, r, radian_ls[1], radian_ls[2], 0.001)
    social_segment_2 = get_segment_coords(h, k, r, radian_ls[3], radian_ls[4], 0.001)
    social_segment_3 = get_segment_coords(h, k, r, radian_ls[5], radian_ls[6], 0.001)
    social_segment_4_1 = get_segment_coords(h, k, r, radian_ls[7], radian_ls[8], 0.001)
    social_segment_4_2 = get_segment_coords(h, k, r, radian_ls[9], radian_ls[0], 0.001)

    xy_coords = social_segment_1 + social_segment_3 + social_segment_2 + social_segment_4_1 + social_segment_4_2

    return xy_coords


def draw_circle(x, y, img, radius = 1):
    jj, kk = circle_perimeter(int(y), int(x), 1)
    img[jj, kk] = 255


def triangle_type(a, b, c):

    if not (a + b > c and a + c > b and b + c > a):
        return "invalid triangle"

    longest_side = max(a, b, c)
    other_sides = [x for x in [a, b, c] if x != longest_side]

    if longest_side ** 2 > sum(x ** 2 for x in other_sides):
        return "obtuse"
    elif longest_side ** 2 < sum(x ** 2 for x in other_sides):
        return "acute"
    else:
        return "right"


def heading_vector():




class Larvae():
    def __init__(self, frame, label_obj_property_dict):
        self.frame = frame

        # Centroid of the larvae
        self.frame_y_pixel, self.frame_x_pixel = label_obj_property_dict.centroid

        # Proportion of pixels in a well for x and y dimensions (for conversions)
        self.well_pixel_prop_x = self.frame.n_pixels_x/self.frame.n_well_cols
        self.well_pixel_prop_y = self.frame.n_pixels_y/self.frame.n_well_rows

        # Get larvae index
        self.row_i = int(self.frame_y_pixel /  self.well_pixel_prop_y)
        self.col_i = int(self.frame_x_pixel /  self.well_pixel_prop_x)
        self.larvae_i = (self.row_i * frame.n_well_cols) + self.col_i

        # Making the frame/larvae identifier string
        s = '00000'
        f = str(self.frame.frame_i)[::-1]
        frame_str = ''.join([f[i] if i < len(f) else s[i] for i in range(len(s))][::-1])
        l = str(self.larvae_i)[::-1]
        larvae_str = ''.join([l[i] if i < len(l) else s[i] for i in range(len(s[:2]))][::-1])

        # Variables for parameter analysis
        self.frame_larvae_i = 'fr'+frame_str+'_larv'+larvae_str

        self.tracking_method = None

        self.well_pt1 = 'NaN'
        self.well_pt2 = 'NaN'
        self.well_pt3 = 'NaN'

        self.pt1_angle = 'NaN'
        self.pt2_angle = 'NaN'
        self.pt3_angle = 'NaN'

        self.edge1 = 'NaN'
        self.edge2 = 'NaN'
        self.edge3 = 'NaN'

        self.tri_area = 'NaN'
        self.heading_angle = 'NaN'

        # Cropped well images
        self.sharp_well_img = self.crop_well_img(self.frame.sharp_frame_img)
        self.sharp_subtracted_well_img = self.crop_well_img(self.frame.sharp_subtracted_img)
        self.low_thresholded_well_img = self.crop_well_img(self.frame.low_thresholded_img)

        self.track_3pt_img_property_dicts = None
        self.track_2pt_img_property_dicts = None
        self.track_1pt_img_property_dicts = None



        # Directories for well images
        self.expanded_img_dir = self.frame.img_dir+'/larvae'+str(self.larvae_i)+'/expanded_img_dir/'
        os.makedirs(os.path.dirname(self.expanded_img_dir), exist_ok=True)
        self.widened_img_dir = self.frame.img_dir+'/larvae'+str(self.larvae_i)+'/widened_img_dir/'
        os.makedirs(os.path.dirname(self.widened_img_dir), exist_ok=True)
        self.track_img_dir = self.frame.img_dir+'/larvae'+str(self.larvae_i)+'/track_img_dir/'
        os.makedirs(os.path.dirname(self.track_img_dir), exist_ok=True)
        self.well_img_dir = self.frame.img_dir+'/larvae'+str(self.larvae_i)+'/well_imgs/'
        os.makedirs(os.path.dirname(self.well_img_dir), exist_ok=True)
        self.low_thresh_img_dir = self.frame.img_dir+'/larvae'+str(self.larvae_i)+'/low_thresh_imgs/'
        os.makedirs(os.path.dirname(self.low_thresh_img_dir), exist_ok=True)

        cv2.imwrite(self.low_thresh_img_dir+'cent_image_larvae'+str(self.larvae_i)+'_frame'+str(self.frame.frame_i)+'.png', self.low_thresholded_well_img)


    def frame_to_well_values(self, x_frame_value, y_frame_value):
        x_well_value = x_frame_value - (self.col_i*self.well_pixel_prop_x)
        y_well_value = y_frame_value - (self.row_i* self.well_pixel_prop_y)
        return x_well_value, y_well_value


    def well_to_frame_values(self, x_well_value, y_well_value):
        x_frame_value = (self.col_i*self.well_pixel_prop_x) + x_well_value
        y_frame_value = (self.row_i* self.well_pixel_prop_y) + y_well_value
        return x_frame_value, y_frame_value


    def crop_well_img(self, image_type):
            x0 = int(self.well_pixel_prop_x) * self.col_i
            x1 = int(self.well_pixel_prop_x) * (self.col_i + 1)
            y0 = int(self.well_pixel_prop_y) * self.row_i
            y1 = int(self.well_pixel_prop_y) * (self.row_i + 1)

            larvae_well_img = image_type[y0:y1, x0:x1]

            return larvae_well_img


    def assign_vars(self, points, edges, angles, ant_triangle=False, ant_segment=False):

        if self.tracking_method == '3pt':

            p1, p2, p3 = points
            e1, e2, e3 = edges
            a1, a2, a3 = angles

            self.well_pt1 = tuple(p1)
            self.well_pt2 = tuple(p2)
            self.well_pt3 = tuple(p3)

            self.edge1 = e1 * self.frame.pixel_um
            self.edge2 = e2 * self.frame.pixel_um
            self.edge3 = e3 * self.frame.pixel_um

            self.pt1_angle = a1
            self.pt2_angle = a2
            self.pt3_angle = a3

            self.sharp_well_img[int(p1[1]), int(p1[0])] = 255
            self.sharp_well_img[int(p2[1]), int(p2[0])] = 255
            self.sharp_well_img[int(p3[1]), int(p3[0])] = 255

            self.tri_area = self.tri_area * (self.frame.pixel_um**2)

            # determine which are eye poitna dn which is swim bladder
            # use the bisector
            self.heading_angle = heading_vector(self.tracking_method, points, )

        if self.tracking_method == '2pt':

            p1, p2 = points
            self.well_pt1 = tuple(p1)
            self.well_pt2 = tuple(p2)

            self.edge1 = euclidean(list(map(float, self.well_pt1)),list(map(float, self.well_pt2))) * self.frame.pixel_um

            self.sharp_well_img[int(p1[1]), int(p1[0])] = 255
            self.sharp_well_img[int(p2[1]), int(p2[0])] = 255
            self.heading_angle = heading_vector()


        if self.tracking_method == '1pt':

            p1 = points
            self.well_pt1 = tuple(p1)
            self.sharp_well_img[int(p1[1]), int(p1[0])] = 255

            # counter for frame sequences with 1 point tracked, once 2pt or 3pt trackign resumes, interpolate heading angles in this sequence


    def track_3pt(self, labeled_img):
        self.track_3pt_img_property_dicts = regionprops(labeled_img)
        [pt1, pt2, pt3] = [Point(self.track_3pt_img_property_dicts[0].centroid[1], self.track_3pt_img_property_dicts[0].centroid[0]),
                           Point(self.track_3pt_img_property_dicts[1].centroid[1], self.track_3pt_img_property_dicts[1].centroid[0]),
                           Point(self.track_3pt_img_property_dicts[2].centroid[1], self.track_3pt_img_property_dicts[2].centroid[0])]

        try:
            triangle = Triangle(pt1, pt2, pt3)
            self.tri_area = abs(float(triangle.area))

        except AttributeError:
            self.tracking_method = '2pt'

            return

        [edge1_len, edge2_len, edge3_len] = sorted([euclidean(list(map(float, pt1)),list(map(float, pt2))),
                                                    euclidean(list(map(float, pt2)),list(map(float, pt3))),
                                                    euclidean(list(map(float, pt3)),list(map(float, pt1)))])
        tri_type = triangle_type(edge1_len, edge2_len, edge3_len)

        if tri_type == 'acute':

            if self.tri_area > 6.0:                     # passes most false positives

                self.tracking_method = '3pt'
                self.assign_vars(points=[pt1, pt2, pt3],
                                 edges=[edge1_len, edge2_len, edge3_len],
                                 angles=[triangle.angles[pt1], triangle.angles[pt2], triangle.angles[pt3]],
                                 ant_triangle=triangle)

                return 'positive'                       # captures most true positives
            else:
                self.tracking_method = '2pt'
                return 'negative'

        if tri_type == 'right':

            self.tracking_method = '3pt'
            self.assign_vars(points=[pt1, pt2, pt3],
                             edges=[edge1_len, edge2_len, edge3_len],
                             angles=[triangle.angles[pt1], triangle.angles[pt2], triangle.angles[pt3]],
                             ant_triangle=triangle)

            return 'positive'                           # captures some true positives

        shortest_edge = min([float(edge1_len), float(edge2_len), float(edge3_len)])
        [p1_angle, p2_angle, p3_angle] = [triangle.angles[pt1],triangle.angles[pt2],triangle.angles[pt3]]
        largest_angle = max([p1_angle, p2_angle, p3_angle])

        if tri_type == 'obtuse':

            if self.tri_area > 11.0:
                if shortest_edge > 5.0:                 # captures most false negatives

                    self.tracking_method = '2pt'
                    return 'negative'                   # passes true negatives
                else:

                    self.tracking_method = '3pt'
                    self.assign_vars(points=[pt1, pt2, pt3],
                                     edges=[edge1_len, edge2_len, edge3_len],
                                     angles=[triangle.angles[pt1], triangle.angles[pt2], triangle.angles[pt3]],
                                     ant_triangle=triangle)

                    return 'positive'                   # captures most false negatives

            if largest_angle < 1.7:
                self.tracking_method = '3pt'
                self.assign_vars(points=[pt1, pt2, pt3],
                                 edges=[edge1_len, edge2_len, edge3_len],
                                 angles=[triangle.angles[pt1], triangle.angles[pt2], triangle.angles[pt3]],
                                 ant_triangle=triangle)

                return 'positive'

            else:
                self.tracking_method = '2pt'
                return 'negative'


    def track_2pt(self, labeled_img):
        self.track_2pt_img_property_dicts = regionprops(labeled_img)
        [pt1, pt2] = [Point(self.track_2pt_img_property_dicts[0].centroid[1], self.track_2pt_img_property_dicts[0].centroid[0]),
                      Point(self.track_2pt_img_property_dicts[1].centroid[1], self.track_2pt_img_property_dicts[1].centroid[0])]
        self.tracking_method = '2pt'
        self.assign_vars(points=[pt1, pt2], ant_segment=)

        return 'positive'


    def track_1pt(self, labeled_img):
        self.track_1pt_img_property_dicts = regionprops(labeled_img)
        if len(self.track_1pt_img_property_dicts) > 1:
            object_areas = [self.track_1pt_img_property_dicts[i].area for i in range(len(self.track_1pt_img_property_dicts))]
            max_index = object_areas.index(max(object_areas))
            pt1 = Point(self.track_1pt_img_property_dicts[max_index].centroid[1], self.track_1pt_img_property_dicts[max_index].centroid[0])
        else:
            pt1 = Point(self.track_1pt_img_property_dicts[0].centroid[1], self.track_1pt_img_property_dicts[0].centroid[0])
        self.tracking_method = '1pt'
        self.assign_vars(pt1, '', '')


    def amplitude_metrics(self):
        sharp_subtracted_well_img = self.sharp_subtracted_well_img
        # Expand distribution of pixel intensities
        expanded_img = sharp_subtracted_well_img * 4        # play with later
        expanded_img = expanded_img - 70                    # or 80, play with later
        expanded_img[expanded_img < 0] = 0
        expanded_thresholded_img = cv2.threshold(expanded_img, 2, 255, cv2.THRESH_BINARY)[1]
        expanded_labeled_img = label(expanded_thresholded_img)
        n_objects = np.amax(expanded_labeled_img)
        expanded_img_property_dicts = regionprops(expanded_labeled_img)
        obj_areas = [expanded_img_property_dicts[i].area for i in range(n_objects)]
        obj_areas_max_i = obj_areas.index(max(obj_areas))

        # Flatten / widen distribution of pixel intensities
        larvae_pixel_coords = expanded_img_property_dicts[obj_areas_max_i].coords
        larvae_pixel_ints = sharp_subtracted_well_img[larvae_pixel_coords[:, 0], larvae_pixel_coords[:, 1]]
        larvae_mean = np.mean(larvae_pixel_ints)
        larvae_std_dev = np.std(larvae_pixel_ints)
        new_std_dev = larvae_std_dev * 3
        centered_data = larvae_pixel_ints - larvae_mean
        scaled_data = centered_data * (new_std_dev / larvae_std_dev)
        widened_data = scaled_data + larvae_mean
        widened_img = np.zeros((102,102))
        widened_img[larvae_pixel_coords[:, 0], larvae_pixel_coords[:, 1]] = widened_data

        min_thresh = 40         # minimum threshold of eye_img to consider without going as low as the low_threshold_img
        max_thresh = 255        # maximum threshold is the maximum intensity of the pixels in the image

        thresholds = sorted(range(min_thresh,max_thresh), reverse=True)

        # track 3 points if you can (2 eyes and swimbladder), if formulation of them is not correct,
        # track 2 points (center of the eyes and swimbladder), if this is also not correct or not possible,
        # track 1 point (centroid of reduced head/pectoral girdle complex)

        for threshold in thresholds:

            track_img = cv2.threshold(widened_img, threshold, 255, cv2.THRESH_BINARY)[1]
            track_labeled_img = label(track_img)
            track_3pt_n_objects = np.amax(track_labeled_img)

            if track_3pt_n_objects == 3:

                if self.track_3pt(track_labeled_img) == 'positive':
                    cv2.imwrite(self.track_img_dir + 'cent_image_larvae' + str(self.larvae_i) + '_frame' + str(self.frame.frame_i) + '.png', track_img)

                    cv2.imwrite(self.expanded_img_dir + 'cent_image_larvae' + str(self.larvae_i) + '_frame' + str(self.frame.frame_i) + '.png', expanded_img)

                    cv2.imwrite(self.widened_img_dir + 'cent_image_larvae' + str(self.larvae_i) + '_frame' + str(self.frame.frame_i) + '.png', widened_img)

                    cv2.imwrite(self.well_img_dir + 'well_image_larvae' + str(self.larvae_i) + '_frame' + str(self.frame.frame_i) + '.png', self.sharp_well_img)
                    return
                if self.track_3pt(track_labeled_img) == 'negative':
                    break

            else: continue

            if track_3pt_n_objects > 3: break

        track_img = cv2.threshold(widened_img, 80, 255, cv2.THRESH_BINARY)[1]
        track_labeled_img = label(track_img)
        track_2pt_n_objects = np.amax(track_labeled_img)

        if track_2pt_n_objects == 2:

            if self.track_2pt(track_labeled_img) == 'positive':

                cv2.imwrite(self.track_img_dir + 'cent_image_larvae' + str(self.larvae_i) + '_frame' + str(self.frame.frame_i) + '.png',track_img)

                cv2.imwrite(self.expanded_img_dir + 'cent_image_larvae' + str(self.larvae_i) + '_frame' + str(self.frame.frame_i) + '.png', expanded_img)

                cv2.imwrite(self.widened_img_dir + 'cent_image_larvae' + str(self.larvae_i) + '_frame' + str(self.frame.frame_i) + '.png', widened_img)

                cv2.imwrite(self.well_img_dir + 'well_image_larvae' + str(self.larvae_i) + '_frame' + str(self.frame.frame_i) + '.png', self.sharp_well_img)

                return

        track_labeled_img = label(self.low_thresholded_well_img)
        self.track_1pt(track_labeled_img)

        cv2.imwrite(self.track_img_dir + 'cent_image_larvae' + str(self.larvae_i) + '_frame' + str(self.frame.frame_i) + '.png', track_img)

        cv2.imwrite(self.expanded_img_dir + 'cent_image_larvae' + str(self.larvae_i) + '_frame' + str(self.frame.frame_i) + '.png', expanded_img)

        cv2.imwrite(self.widened_img_dir + 'cent_image_larvae' + str(self.larvae_i) + '_frame' + str(self.frame.frame_i) + '.png', widened_img)

        cv2.imwrite(self.well_img_dir + 'well_image_larvae' + str(self.larvae_i) + '_frame' + str(self.frame.frame_i) + '.png', self.sharp_well_img)

            # sharp_well_img = np.float32(self.sharp_well_img)
            # cv2.line(sharp_well_img,(int(self.pt1_well_x_pixel), int(pt1_well_y_pixel))(int(pt2_well_x_pixel), int(pt2_well_y_pixel)),(255, 0, 0), 1)
            # self.sharp_well_img = sharp_well_img



        return

            # factor in proximity to the wall during the event?
            # interpolate between lost points and annotate interpolated points
            # try to use lower thesholded image to get the farthest point of the tail possible
            # goal is to use the axis of symmetry between the 2 eyes to the centroid with the approach used in Owen;s code
            # to get the bend angle of the spine from the skull to the end of the tail
            # Owen;s search algorithm uses a set distance from the origin in a specific direction, and a characterized arc
            # at that point to search along in order to find the point with the highest intensity (assumed to be the pint above the spine)
            # there's no max bend in the parameters??
