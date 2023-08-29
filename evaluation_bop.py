

"""
See https://bop.felk.cvut.cz/challenges/bop-challenge-2019/#howtoparticipate
"""


import time

from olt.tracker import Tracker
from olt.localizer import Localizer
from olt.evaluation_tools import BOPDatasetReader, append_result, run_bop_evaluation
from olt.config import BOP_DS_DIRS, OBJ_MODEL_DIRS, EvaluationBOPConfig
from olt.utils import create_video_from_images, obj_name2id

from bop_toolkit_lib import inout  # noqa


eval_cfg = EvaluationBOPConfig() 
eval_cfg.ds_name = 'ycbv'

eval_cfg.tracker_cfg.viewer_display = False
eval_cfg.tracker_cfg.viewer_save = False
eval_cfg.tracker_cfg.n_corr_iterations = 5
eval_cfg.tracker_cfg.n_update_iterations = 2
# eval_cfg.tracker_cfg.n_corr_iterations = 0
# eval_cfg.tracker_cfg.n_update_iterations = 0
eval_cfg.tracker_cfg.tikhonov_parameter_rotation = 10000.0
eval_cfg.tracker_cfg.tikhonov_parameter_translation = 30000.0

SKIP_N_IMAGES = 0
NB_IMG_RUN = -1  # all
# NB_IMG_RUN = 20

# Run the pose estimation (or get the groundtruth) every <LOCALIZE_EVERY> frames
LOCALIZE_EVERY = NB_IMG_RUN # Never again
# LOCALIZE_EVERY = 1
LOCALIZE_EVERY = 10

PRINT_INFO_EVERY = 60


ds_name = eval_cfg.ds_name
# ds_name = 'rotd'
reader = BOPDatasetReader(ds_name)

eval_cfg.localizer_cfg.n_workers = 2
localizer = Localizer(eval_cfg.ds_name, eval_cfg.localizer_cfg)

all_sids = sorted(reader.map_sids_vids.keys())
all_results = []


for sid in all_sids:
    print(f"New scene: {sid}")
    vid0 = reader.map_sids_vids[sid][0]

    accepted_objs='all'
    tracker = Tracker(reader.get_intrinsics(sid, vid0), OBJ_MODEL_DIRS[eval_cfg.ds_name], accepted_objs, eval_cfg.tracker_cfg)
    tracker.init()

    K, height, width = reader.get_cam_data(sid, vid0)
    rgb = reader.get_img(sid, vid0)

    # poses = localizer.predict(rgb, reader.K, n_coarse=1, n_refiner=1)
    # poses = reader.predict_gt(sid=sid, vid=vid0)

    N_views = len(reader.map_sids_vids[sid])
    print('N_views: ', N_views)
    for i in range(N_views):
        if i < SKIP_N_IMAGES:
            continue
        
        vid = reader.map_sids_vids[sid][i]

        K, height, width = reader.get_cam_data(sid, vid)
        rgb = reader.get_img(sid, vid)

        if i % LOCALIZE_EVERY == 0:
            # poses = localizer.predict(rgb, reader.K, n_coarse=1, n_refiner=1)
            poses = reader.predict_gt(sid=sid, vid=vid)
            # print(poses)

        tracker.detected_bodies(poses)
        tracker.set_image(rgb)
        dt_track = tracker.track()
        t = time.time()
        if eval_cfg.tracker_cfg.viewer_display or eval_cfg.tracker_cfg.viewer_save:
            tracker.update_viewers()
        if i % PRINT_INFO_EVERY == 0:
            print(f'View: {i}/{N_views}')
            print('track() (ms)', 1000*dt_track)
            print('update_viewers() (ms)', 1000*(time.time() - t))

        # TODO:
        # Get current tracks from tracker -> new method!
        tracker_preds = tracker.get_current_preds()


        # scene_id, obj_id, view_id, score, TCO, dt
        score = 1
        dt = 0.5
        if reader.check_if_in_bop19_targets(sid, vid):
            for obj_name, TCO in tracker_preds.items():
                append_result(all_results, sid, obj_name2id(obj_name), vid, score, TCO, dt)

        
        if i == NB_IMG_RUN:
            break


# bop result file name stricly formatted:<method>_<ds_name>-<split>.csv
result_bop_eval_filename = 'tracker_ycbv-test.csv'
inout.save_bop_results(f'results/{result_bop_eval_filename}', all_results)
run_bop_evaluation(result_bop_eval_filename)

vid_name = f'result_{eval_cfg.ds_name}_{sid}.mp4'
# create_video_from_images(tracker.imgs_dir, vid_name=vid_name)