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
        self.video_contrast = self.video_obj.set(cv2.CAP_PROP_CONTRAST, 100)

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

        self.video_frames = []
        self.video_frame_larve_df = self.process_video()

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
        '''
        Use the length of the video and the number of preffered images to calculate
        a median image to serve as the background image for all the images of the video.
        :return:
        '''
        bg_interval = int(self.video_len/self.num_bg_frames)
        bg_tot_init_arr = np.zeros((self.num_bg_frames, 1024, 1024))
        bg_tot_init_arr[:] = np.nan

        for n_interval in range(self.num_bg_frames):
            frame_i = n_interval*bg_interval
            bg_frame = self.read_frame(frame_i)
            bg_tot_init_arr[n_interval] = bg_frame

        # Calculate the median of the array of selected images and save it
        median_bg = np.median(bg_tot_init_arr, axis=0)
        cv2.imwrite(self.video_fn[:-4]+'background.png', median_bg)

        return median_bg

    def process_video(self):

        frame_df_ls = []

        for frame_i in range(int(self.video_len))[5115:5145]:
            print('frame', frame_i)
            frame = Frame(self, frame_i, self.read_frame(frame_i), self.background_img)
            frame_df_ls.append(frame.frame_larvae_df)
            if frame_i % 100 == 0: print('tracked', self.video_fn, frame_i,'/',int(self.video_len))

        video_frame_larve_df = pd.concat(frame_df_ls)

        return video_frame_larve_df