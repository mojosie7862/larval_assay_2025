import numpy as np
import cv2
import pandas as pd
from skimage.measure import label, regionprops # can add extra property functions - use for larvae length
from skimage.morphology import remove_small_objects
from sklearn.datasets import make_circles

from larvae import Larvae
import os
import matplotlib.pyplot as plt
from skimage.draw import circle_perimeter


def draw_circle(x, y, img, radius = 1):
    jj, kk = circle_perimeter(int(y), int(x), 1)
    img[jj, kk] = 255


class Frame():

    def __init__(self, experiment_video, frame_i, frame_img, background_img):
        self.experiment_video = experiment_video
        self.frame_i = frame_i

        self.sharp_frame_img = frame_img
        self.blurred_frame_img = cv2.GaussianBlur(self.sharp_frame_img, (3, 3), 0)
        self.sharp_background_img = background_img
        self.blurred_background_img = cv2.GaussianBlur(self.sharp_background_img, (3, 3), 0)

        self.sharp_subtracted_img = None
        self.blurred_subtracted_img = None
        self.low_thresholded_img = None

        self.n_larvae = 100
        self.n_well_rows = 10
        self.n_well_cols = 10
        self.n_pixels_x = 1024
        self.n_pixels_y = 1024
        self.frame_t_ms = 1 / self.experiment_video.fps * 1000
        self.pixel_um = 87

        self.larva_xy_columns = ['frame_larvae_i','tracking_method','pt1','pt2','pt3','pt1_angle','pt2_angle','pt3_angle',
                                 'edge1','edge2','edge3','tri_area','heading_angle']
        self.larva_xy_types_dict = {'frame_larvae_i': str,
                                    'tracking_method': str,
                                    'pt1_angle': float,
                                    'pt2_angle': float,
                                    'pt3_angle': float,
                                    'edge1': float,
                                    'edge2': float,
                                    'edge3': float,
                                    'tri_area': float,
                                    # 'larvae_area':float,
                                    'heading_angle': float}

        self.frame_img_property_dicts = None

        self.img_dir = 'testing_images/amp_tracking/frame_imgs+larvae_dirs/'
        self.frame_img_dir = self.img_dir + '/frame_imgs/'
        os.makedirs(os.path.dirname(self.frame_img_dir), exist_ok=True)

        self.frame_larvae_df = self.process_frame()




    def process_frame(self):
        # Set up coordinate dictionary
        larvae_keys = list(range(self.n_larvae))
        larvae_dict = dict.fromkeys(larvae_keys)

        # Subtract the background noise of the frame
        self.sharp_subtracted_img = abs(self.sharp_background_img - self.sharp_frame_img)
        self.blurred_subtracted_img = abs(self.blurred_background_img - self.blurred_frame_img)

        self.low_thresholded_img = cv2.threshold(self.blurred_subtracted_img, 15, 255, cv2.THRESH_BINARY)[1]    # Uses intensity of the pixel value as a threshold to highlight zebrafish silouhettes in white and background in black
        labeled_img = label(self.low_thresholded_img)                                                                         # Label each independent object in the image using integers
        cleaned_img = remove_small_objects(labeled_img, 45)                                                           # Remove noise objects
        relabeled_img = label(cleaned_img)                                                                                    # Relabel each zebrafish

        # PARSE DETECTED LARVAE AND CREATE LARVAE OBJECT

        self.frame_img_property_dicts = regionprops(relabeled_img)
        max_label_obj_index = np.amax(relabeled_img)

        if max_label_obj_index!=100:
            print('low threshold img has', max_label_obj_index, 'objects')


        for label_obj_i in range(max_label_obj_index):

            # LARVAE OBJECT
            larvae = Larvae(self, self.frame_img_property_dicts[label_obj_i])

            larvae.amplitude_metrics()

            larvae_dict[larvae.larvae_i] = (larvae.frame_larvae_i, larvae.tracking_method,
                                            larvae.well_pt1, larvae.well_pt2, larvae.well_pt3,
                                            larvae.pt1_angle, larvae.pt2_angle,larvae.pt3_angle,
                                            larvae.edge1, larvae.edge2, larvae.edge3,
                                            larvae.tri_area, larvae.heading_angle)

            cv2.imwrite(self.frame_img_dir + 'frame' + str(self.frame_i) + '.png', self.sharp_frame_img)


        # Mistracked larvae / empty wells are assigned NaN value
        # some_none = not all(larvae_dict.values())
        # if some_none:
        #     for larvae_i, track_data in larvae_dict.items():
        #         if not track_data:
        #             larvae_dict[larvae_i] = (larvae_i,int(self.frame_i),'NaN','NaN','NaN','NaN')

        # Frame xy dictionary transformed to a pandas dataframe
        larva_xy_df = pd.DataFrame.from_dict(larvae_dict, orient='index', columns=self.larva_xy_columns).astype(self.larva_xy_types_dict)

        return larva_xy_df

