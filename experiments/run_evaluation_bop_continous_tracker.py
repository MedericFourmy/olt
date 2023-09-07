


"""
See https://bop.felk.cvut.cz/challenges/bop-challenge-2019/#howtoparticipate
"""


import time
from pathlib import Path

from olt.continuous_tracker import ContinuousTracker
from olt.evaluation_tools import BOPDatasetReader, append_result, run_bop_evaluation
from olt.config import OBJ_MODEL_DIRS, EvaluationBOPConfig
from olt.utils import create_video_from_images, obj_name2id, intrinsics2Kres
from olt.rate import Rate

from bop_toolkit_lib import inout  # noqa


USE_DEPTH = False 
SKIP_N_IMAGES = 0
NB_IMG_RUN = -1  # all
# NB_IMG_RUN = 50
# Run the pose estimation (or get the groundtruth) every <LOCALIZE_EVERY> frames
# LOCALIZE_EVERY = NB_IMG_RUN # Never again
# LOCALIZE_EVERY = 1
PRINT_INFO_EVERY = 60
# USE_GT_FOR_LOCALIZATION = True
# FAKE_LOCALIZATION_DELAY = 0.5
USE_GT_FOR_LOCALIZATION = False
FAKE_LOCALIZATION_DELAY = 0.0
IMG_FREQ = 30

eval_cfg = EvaluationBOPConfig() 
eval_cfg.ds_name = 'ycbv'

eval_cfg.tracker_cfg.viewer_display = True
eval_cfg.tracker_cfg.viewer_save = False
eval_cfg.tracker_cfg.n_corr_iterations = 4
eval_cfg.tracker_cfg.n_update_iterations = 2
# eval_cfg.tracker_cfg.n_corr_iterations = 0
# eval_cfg.tracker_cfg.n_update_iterations = 0
eval_cfg.tracker_cfg.use_depth = USE_DEPTH
eval_cfg.tracker_cfg.measure_occlusions = USE_DEPTH

# eval_cfg.tracker_cfg.n_corr_iterations = 0
# eval_cfg.tracker_cfg.n_update_iterations = 0
eval_cfg.tracker_cfg.tikhonov_parameter_rotation = 2000.0
eval_cfg.tracker_cfg.tikhonov_parameter_translation = 30000.0

# eval_cfg.tracker_cfg.depth_standard_deviations = [0.0, 0.0, 0.0]
eval_cfg.tracker_cfg.depth_standard_deviations = [0.05, 0.03, 0.02]

eval_cfg.localizer_cfg.detector_threshold = 0.7




ds_name = eval_cfg.ds_name
# ds_name = 'rotd'
reader = BOPDatasetReader(ds_name, load_depth=USE_DEPTH)

eval_cfg.localizer_cfg.n_workers = 1


all_sids = sorted(reader.map_sids_vids.keys())
sidmax = all_sids[-1]
# all_sids = [all_sids[1]]

all_results = []

rate = Rate(IMG_FREQ)

if __name__ == '__main__':


    for sid in all_sids:
        # # HACK: run only first scene
        # if sid > all_sids[0]:
        #     break

        vids = reader.map_sids_vids[sid]

        rgb_intrinsics = reader.get_intrinsics(sid, vids[0])
        depth_intrinsics = rgb_intrinsics if USE_DEPTH else None  # same for YCBV

        continuous_tracker = ContinuousTracker(
            tracker_cfg=eval_cfg.tracker_cfg,
            localizer_cfg=eval_cfg.localizer_cfg,
            ds_name=eval_cfg.ds_name,
            rgb_intrinsics=rgb_intrinsics,
            depth_intrinsics=depth_intrinsics,
            collect_statistics=True,
            fake_localization_delay=FAKE_LOCALIZATION_DELAY
        )

        # initialize
        object_poses = reader.predict_gt(sid, vids[0]) if USE_GT_FOR_LOCALIZATION else None
        obs = reader.get_obs(sid, vids[0])
        depth = obs.depth if USE_DEPTH else None
        continuous_tracker(obs.rgb, depth, object_poses, sid, vids[0])

        # Preload observation for the scene as the is a bit I/O costly
        observations = [reader.get_obs(sid, vid) for vid in vids]

        N_views = len(vids)
        for i in range(N_views):
            if i < SKIP_N_IMAGES:
                continue
            if i == NB_IMG_RUN:
                break
            
            K, height, width = intrinsics2Kres(**rgb_intrinsics)
            obs = observations[i]
            depth = obs.depth if USE_DEPTH else None
            object_poses = reader.predict_gt(sid, vids[i]) if USE_GT_FOR_LOCALIZATION else None

            t = time.perf_counter()
            tracker_preds = continuous_tracker(obs.rgb, depth, object_poses, sid, vids[i])
            print(f'{vids[i]} tracker_preds', tracker_preds)
            dt_track = time.perf_counter() - t

            if i % PRINT_INFO_EVERY == 0:
                print(f'Scene: {sid}/{sidmax}, View: {vids[i]}/{vids[-1]}')
                print('track() (ms)', 1000*dt_track)
                print('update_viewers() (ms)', 1000*(time.perf_counter() - t))

            # scene_id, obj_id, view_id, score, TCO, dt
            score = 1
            dt = dt_track
            if reader.check_if_in_bop19_targets(sid, vids[i]):
                for obj_name, TCO in tracker_preds.items():
                    append_result(all_results, sid, obj_name2id(obj_name), vids[i], score, TCO, dt)

            rate.sleep()



    RESULTS_DIR_NAME = Path('results')
    EVALUATIONS_DIR_NAME = Path('evaluations')
    RESULTS_DIR_NAME.mkdir(exist_ok=True)
    EVALUATIONS_DIR_NAME.mkdir(exist_ok=True)

    # bop result file name stricly formatted:<method>_<ds_name>-<split>.csv
    result_bop_eval_filename = 'tracker_ycbv-test.csv'
    inout.save_bop_results(f'{RESULTS_DIR_NAME.as_posix()}/{result_bop_eval_filename}', all_results)
    run_bop_evaluation(result_bop_eval_filename, RESULTS_DIR_NAME, EVALUATIONS_DIR_NAME)

    vid_name = f'result_{eval_cfg.ds_name}_{sid}.mp4'
    # create_video_from_images(tracker.imgs_dir, vid_name=vid_name)