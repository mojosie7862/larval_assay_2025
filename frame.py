

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