import cv2
import numpy as np
import pandas as pd
from frame import Frame


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