import os
import time
from video import Video
import pandas as pd

fn = '241217-nrxn1a_wd_1a_hetinx-'

# set directory containing image files
video_dir = "E:/jm_laCie_larval_behavior_avi_files/"
save_dir = "E:/jm_laCie_larval_behavior_tracked_data/"

# create directories to save experiment data
bl_exp_dir = save_dir + fn + 'BL_.avi'
df_exp_dir = save_dir + fn + 'DF_.avi'
as_exp_dir = save_dir + fn + 'AS_.avi'

paradigm_dir_ls = [bl_exp_dir, df_exp_dir, as_exp_dir]

for para_dir in paradigm_dir_ls:

    exp_tracking_dir = para_dir + '/'
    exp_tracking_img_dir = para_dir + '/trackedimages/'

    os.makedirs(os.path.dirname(exp_tracking_dir), exist_ok=True)
    os.makedirs(os.path.dirname(exp_tracking_img_dir), exist_ok=True)

def track_experiment(dir, v_fn, para):
    v_obj = Video(dir, v_fn, para)
    return v_obj.larva_xy_df, v_fn

if __name__ == '__main__':

    start = time.perf_counter()

    video_dir_ls = [video_dir, video_dir, video_dir]
    video_fn_ls =[fn + 'BL_.avi', fn + 'DF_.avi', fn + 'AS_.avi']
    paradigm_ls = ['BL', 'DF', 'AS']

    # with cf.ProcessPoolExecutor(max_workers=3) as executor:
    #     for return_df, df_fn in executor.map(track_experiment, video_dir_ls, video_fn_ls, paradigm_ls):
    #         return_df.to_csv(df_fn[:-4]+'xy_tracking.csv')
    #
    # finish = time.perf_counter()
    # print(f'finished in {round(finish-start, 2)} sec')

    AS_exp = Video(video_dir, fn+'AS_.avi', 'AS')

    AS_exp.video_frame_larve_df.to_csv(fn+'AS_track_data'+'_tracking_vars.csv')



