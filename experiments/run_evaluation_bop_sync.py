

"""
See https://bop.felk.cvut.cz/challenges/bop-challenge-2019/#howtoparticipate
"""


import time
import cv2
import numpy as np
from pathlib import Path

from olt.tracker import Tracker
from olt.localizer import Localizer
from olt.evaluation_tools import BOPDatasetReader, append_result, run_bop_evaluation
from olt.config import OBJ_MODEL_DIRS, EvaluationBOPConfig
from olt.utils import create_video_from_images, obj_name2id

from bop_toolkit_lib import inout  # noqa

USE_DEPTH = True 


eval_cfg = EvaluationBOPConfig() 
eval_cfg.ds_name = 'ycbv'

eval_cfg.tracker_cfg.viewer_display = True
eval_cfg.tracker_cfg.viewer_save = False
eval_cfg.tracker_cfg.n_corr_iterations = 4
eval_cfg.tracker_cfg.n_update_iterations = 2
# eval_cfg.tracker_cfg.n_corr_iterations = 0
# eval_cfg.tracker_cfg.n_update_iterations = 0
eval_cfg.tracker_cfg.use_depth = USE_DEPTH
eval_cfg.tracker_cfg.region_modality.measure_occlusions = True
eval_cfg.tracker_cfg.depth_modality.measure_occlusions = True

# eval_cfg.tracker_cfg.n_corr_iterations = 0
# eval_cfg.tracker_cfg.n_update_iterations = 0
eval_cfg.tracker_cfg.tikhonov_parameter_rotation = 2000.0
eval_cfg.tracker_cfg.tikhonov_parameter_translation = 30000.0

# eval_cfg.tracker_cfg.depth_standard_deviations = [0.0, 0.0, 0.0]
eval_cfg.tracker_cfg.depth_standard_deviations = [0.05, 0.03, 0.02]

eval_cfg.localizer_cfg.detector_threshold = 0.6


SKIP_N_IMAGES = 0


NB_IMG_RUN = -1  # all
# NB_IMG_RUN = 50

# Run the pose estimation (or get the groundtruth) every <LOCALIZE_EVERY> frames
# LOCALIZE_EVERY = NB_IMG_RUN # Never again
# LOCALIZE_EVERY = 1
LOCALIZE_EVERY = 60

PRINT_INFO_EVERY = 60

USE_GT_FOR_LOCALIZATION = False

ds_name = eval_cfg.ds_name
# ds_name = 'rotd'
reader = BOPDatasetReader(ds_name, load_depth=USE_DEPTH)

eval_cfg.localizer_cfg.n_workers = 1
localizer = Localizer(eval_cfg.ds_name, eval_cfg.localizer_cfg)

all_sids = sorted(reader.map_sids_vids.keys())
sidmax = all_sids[-1]

all_sids = [all_sids[1]]

all_bop19_results = []

for sid in all_sids:
    # # HACK: run only first scene
    # if sid > all_sids[0]:
    #     break

    vid0 = reader.map_sids_vids[sid][0]
    vidmax = reader.map_sids_vids[sid][-1]

    # TODO: get it from BOP dataset target (localization taks: we can assume we know which objects are inthe scene)
    accepted_objs='all'
    intrinsics = reader.get_intrinsics(sid, vid0)  # for BOP, rgb and depth have same intrinsics
    color2depth_pose = np.eye(4)
    if USE_DEPTH:
        tracker = Tracker(OBJ_MODEL_DIRS[eval_cfg.ds_name], accepted_objs, eval_cfg.tracker_cfg, intrinsics, intrinsics, color2depth_pose)
    else:
        tracker = Tracker(OBJ_MODEL_DIRS[eval_cfg.ds_name], accepted_objs, eval_cfg.tracker_cfg, intrinsics)
    tracker.init()

    N_views = len(reader.map_sids_vids[sid])
    for i in range(N_views):
        if i < SKIP_N_IMAGES:
            continue
        if i == NB_IMG_RUN:
            break
        
        vid = reader.map_sids_vids[sid][i]

        K, height, width = reader.get_Kres(sid, vid)
        obs = reader.get_obs(sid, vid)

        # # Check RGB vs BGR ordering -> gives unsuable results from CosyPose and does not change the Tracker behavior
        # gray = cv2.cvtColor(obs.rgb, cv2.COLOR_BGR2RGB)

        # # Grayscale for fun -> pretty bad results for both
        # gray = cv2.cvtColor(obs.rgb, cv2.COLOR_BGR2GRAY)
        # obs.rgb = np.dstack([gray]*3)

        #######################################
        # Synchronous localization and tracking
        #######################################
        
        dt_localize = 0
        if i % LOCALIZE_EVERY == 0:
            t = time.perf_counter()
            if USE_GT_FOR_LOCALIZATION:
                poses = reader.predict_gt(sid=sid, vid=vid)
            else:
                poses, scores = localizer.predict(obs.rgb, K, n_coarse=1, n_refiner=6)
            tracker.detected_bodies(poses)
            dt_localize += time.perf_counter() - t
            print('Localise (ms)', 1000*dt_localize)

        if USE_DEPTH:
            tracker.set_image(obs.rgb, obs.depth)
        else:
            tracker.set_image(obs.rgb)
        dt_track = tracker.track()
        t = time.perf_counter()
        if eval_cfg.tracker_cfg.viewer_display or eval_cfg.tracker_cfg.viewer_save:
            tracker.update_viewers()

        if i % PRINT_INFO_EVERY == 0:
            print(f'Scene: {sid}/{sidmax}, View: {vid}/{vidmax}')
            print('track() (ms)', 1000*dt_track)
            print('update_viewers() (ms)', 1000*(time.perf_counter() - t))

        # TODO:
        # Get current tracks from tracker -> new method!
        tracker_preds = tracker.get_current_preds()

        # scene_id, obj_id, view_id, score, TCO, dt
        score = 1
        dt = dt_localize + dt_track
        if reader.check_if_in_targets(sid, vid):
            for obj_name, TCO in tracker_preds.items():
                append_result(all_bop19_results, sid, obj_name2id(obj_name), vid, score, TCO, dt)

        



RESULTS_DIR_NAME = Path('results')
EVALUATIONS_DIR_NAME = Path('evaluations')
RESULTS_DIR_NAME.mkdir(exist_ok=True)
EVALUATIONS_DIR_NAME.mkdir(exist_ok=True)

# bop result file name stricly formatted:<method>_<ds_name>-<split>.csv
result_bop_eval_filename = 'tracker_ycbv-test.csv'
inout.save_bop_results(f'{RESULTS_DIR_NAME.as_posix()}/{result_bop_eval_filename}', all_bop19_results)
run_bop_evaluation(result_bop_eval_filename, RESULTS_DIR_NAME, EVALUATIONS_DIR_NAME)

vid_name = f'result_{eval_cfg.ds_name}_{sid}.mp4'
# create_video_from_images(tracker.imgs_dir, vid_name=vid_name)