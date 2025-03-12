import numpy as np
import cv2
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects


class Video():

    def __init__(self, video_dir, video_fn, paradigm):
        self.paradigm = paradigm
        self.video_path = video_dir + video_fn
        self.video_fn = video_fn
        self.video_obj = self.read_video()
        self.video_len = self.video_obj.get(cv2.CAP_PROP_FRAME_COUNT)

        self.num_bg_frames = 5
        self.background_img = self.extract_background()

        if self.paradigm == 'BL':
            self.n_bocks = 16
            self.fps = 20
        if self.paradigm == 'DF':
            self.n_bocks = 3
            self.fps = 500
        if self.paradigm == 'AS':
            self.n_bocks = 10
            self.fps = 500

        self.minute_dict = {}
        self.event_dict = {}

        self.larva_xy_df = self.track_video()

    def read_frame(self, frame_ind):
        self.video_obj.set(cv2.CAP_PROP_POS_FRAMES,frame_ind)
        ret, frame = self.video_obj.read()
        if ret:
            return frame[:,:,0]
        else:
            print('error reading frame')

    def read_video(self):
        video_obj = cv2.VideoCapture(self.video_path)
        return video_obj

    def extract_background(self):
        bg_interval = int(self.video_len/self.num_bg_frames)
        bg_tot_init_arr = np.zeros((self.num_bg_frames, 1024, 1024))
        bg_tot_init_arr[:] = np.nan

        for n_interval in range(self.num_bg_frames):
            frame_i = n_interval*bg_interval
            bg_frame = self.read_frame(frame_i)
            bg_tot_init_arr[n_interval] = bg_frame

        median_bg = np.median(bg_tot_init_arr, axis=0)
        blurred_bg = cv2.GaussianBlur(median_bg.astype(np.float64), (3, 3), 0)
        cv2.imwrite(self.video_fn[:-4]+'background.png', blurred_bg)
        return blurred_bg

    def track_video(self):
        larva_xy_df_ls = []

        for frame_i in range(int(self.video_len))[:3]:
            frame = Frame(self, frame_i, self.read_frame(frame_i), self.background_img)
            larva_xy_df_ls.append(frame.larva_xy_df)
            if frame_i % 100 == 0: print('tracked', self.video_fn, frame_i,'/',int(self.video_len))

        larva_xy_df = pd.concat(larva_xy_df_ls)

        return larva_xy_df


class Block():

    def __init__(self):
        self.block_secs = 60


class Frame():

    def __init__(self, experiment_video, frame_i, img, background_img):
        self.experiment_video = experiment_video
        self.frame_i = frame_i
        self.img = img
        self.background_img = background_img
        self.thresholded_img = None

        self.n_larvae = 100
        self.n_well_rows = 10
        self.n_well_cols = 10
        self.n_pixels = 1024
        self.frame_t_ms = 1 / self.experiment_video.fps * 1000
        self.pixel_um = 87

        self.larva_xy_columns = ['larvae_i', 'frame_i', 'x_pixel', 'y_pixel', 'x_um', 'y_um']
        self.larva_xy_types_dict = {'larvae_i': int,
                                    'frame_i': int,
                                    'x_pixel': float,
                                    'y_pixel': float,
                                    'x_um': float,
                                    'y_um': float}

        self.larva_xy_df = self.track_frame()
        self.img_property_dict = None


    def track_frame(self):
        # Set up coordinate dictionary
        larva_xy_keys = list(range(100))
        larva_xy_dict = dict.fromkeys(larva_xy_keys)

        # Do a background subtraction, and blur the background
        subtracted_img = abs(self.background_img - self.img)

        # Blur subtracted image
        blurred_img = cv2.GaussianBlur(subtracted_img, (3, 3), 0)

        # Create an image using intensity of the pixel as a threshold of to highlight zebrafish silouhettes in white
        self.thresholded_img = cv2.threshold(blurred_img, 15, 255, cv2.THRESH_BINARY)[1]

        # Label each independent object in the image using integers
        labeled_img = label(self.thresholded_img)

        # Remove noise objects
        cleaned_img = remove_small_objects(labeled_img, 40)

        # Relabel each zebrafish
        relabeled_img = label(cleaned_img)

        # Parse larvae, assign indicies, get coordinates
        self.img_property_dict = regionprops(relabeled_img)
        total_objects = np.amax(relabeled_img)
        for object_index in range(total_objects):

            larvae = Larvae(self, self.img_property_dict[object_index])
            larva_xy_dict[larvae.larvae_i] = (int(larvae.larvae_i), int(self.frame_i), larvae.x_pixel, larvae.y_pixel, larvae.x_um, larvae.y_um)

        # Mistracked larvae / empty wells are assigned NaN value
        some_none = not all(larva_xy_dict.values())
        if some_none:
            for larvae_i, track_data in larva_xy_dict.items():
                if not track_data:
                    larva_xy_dict[larvae_i] = (larvae_i,int(self.frame_i),'NaN','NaN','NaN','NaN')

        # Frame xy dictionary transformed to a pandas dataframe
        larva_xy_df = pd.DataFrame.from_dict(larva_xy_dict, orient='index', columns=self.larva_xy_columns).astype(self.larva_xy_types_dict)

        return larva_xy_df


class Larvae():
    def __init__(self, frame, img_property_dict):
        self.frame = frame
        self.centroid = img_property_dict.centroid
        self.coords = img_property_dict.coords
        self.row_i = int(self.centroid[0] * frame.n_well_rows / frame.n_pixels)
        self.col_i = int(self.centroid[1] * frame.n_well_cols / frame.n_pixels)
        self.larvae_i = (self.row_i * frame.n_well_cols) + self.col_i
        self.y_pixel, self.x_pixel = self.centroid
        self.x_pixel_cropped = self.x_pixel / (self.col_i + 1)
        self.y_pixel_cropped = self.y_pixel / (self.row_i + 1)
        self.x_um = self.x_pixel * self.frame.pixel_um
        self.y_um = self.y_pixel * self.frame.pixel_um
        self.x_um_cropped = self.x_um / (self.col_i + 1)
        self.y_um_cropped = self.y_um / (self.row_i + 1)

        self.larvae_img = self.crop_larvae_img()

    def crop_larvae_img(self):
        x0 = int(self.frame.n_pixels / self.frame.n_well_rows) * self.col_i
        x1 = int(self.frame.n_pixels / self.frame.n_well_rows) * (self.col_i + 1)
        y0 = int(self.frame.n_pixels / self.frame.n_well_rows) * self.row_i
        y1 = int(self.frame.n_pixels / self.frame.n_well_rows) * (self.row_i + 1)

        larvae_img = self.frame.thresholded_img[y0:y1, x0:x1]

        return larvae_img