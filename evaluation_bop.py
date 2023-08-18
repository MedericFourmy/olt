

"""
See https://bop.felk.cvut.cz/challenges/bop-challenge-2019/#howtoparticipate
"""


import time

import resource
print_mem_usg = lambda : print(f'Memory usage: {int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000} (Mb)')

from olt.tracker import Tracker
from olt.localizer import Localizer
from olt.evaluation_tools import BOPDatasetReader
from olt.config import BOP_DS_DIRS, OBJ_MODEL_DIRS, EvaluationBOPConfig
from olt.utils import create_video_from_images


# SCENE_ID = 48  # bad local min for black object
SCENE_ID = 49

eval_cfg = EvaluationBOPConfig() 
eval_cfg.ds_name = 'ycbv'

eval_cfg.tracker_cfg.viewer_display = True
eval_cfg.tracker_cfg.viewer_save = True
eval_cfg.tracker_cfg.n_corr_iterations = 5
eval_cfg.tracker_cfg.n_update_iterations = 2


NB_IMG_RUN = -1  # all
# NB_IMG_RUN = 50

# Run the pose estimation (or get the groundtruth) every <LOCALIZE_EVERY> frames
LOCALIZE_EVERY = NB_IMG_RUN # Never again
LOCALIZE_EVERY = 1
# LOCALIZE_EVERY = 120



reader = BOPDatasetReader(eval_cfg.ds_name)
vid0 = reader.map_sids_vids[SCENE_ID][0]

eval_cfg.localizer_cfg.n_workers = 2
localizer = Localizer(eval_cfg.ds_name, eval_cfg.localizer_cfg)

# accepted_objs = {
#     'obj_000010',
#     'obj_000016',
# }
# accepted_objs = {'obj_000002', 'obj_000004', 'obj_000005', 'obj_000010', 'obj_000015'}
accepted_objs='all'
tracker = Tracker(reader.get_intrinsics(SCENE_ID, vid0), OBJ_MODEL_DIRS[eval_cfg.ds_name], accepted_objs, eval_cfg.tracker_cfg)
tracker.init()


K, height, width = reader.get_cam_data(SCENE_ID, vid0)
rgb = reader.get_img(SCENE_ID, vid0)

# print('torch warming up...')
# poses = localizer.predict(rgb, reader.K, n_coarse=1, n_refiner=1)
# print('... torch warmed up!')
poses = reader.predict_gt(sid=SCENE_ID, vid=vid0)


LOCALIZE_EVERY = 1

for i, vid in enumerate(reader.map_sids_vids[SCENE_ID]):
    K, height, width = reader.get_cam_data(SCENE_ID, vid)
    rgb = reader.get_img(SCENE_ID, vid)

    if i % LOCALIZE_EVERY == 0:
        # poses = localizer.predict(rgb, reader.K, n_coarse=1, n_refiner=1)
        poses = reader.predict_gt(sid=SCENE_ID, vid=vid)

    tracker.detected_bodies(poses)
    tracker.set_image(rgb)
    tracker.track()
    t = time.time()
    tracker.update_viewers()
    print('update_viewers (ms)', 1000*(time.time() - t))

    if i == NB_IMG_RUN:
        break



create_video_from_images(tracker.imgs_dir)