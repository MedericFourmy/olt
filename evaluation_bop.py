

"""
See https://bop.felk.cvut.cz/challenges/bop-challenge-2019/#howtoparticipate
"""


import cv2
import time

import resource
print_mem_usg = lambda : print(f'Memory usage: {int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000} (Mb)')

from olt.tracker import Tracker
from olt.localizer import Localizer
from olt.evaluation_tools import BOPDataReader
from olt.config import BOP_DS_DIRS, OBJECT_MODEL_DIRS, EvaluationBOPConfig


eval_cfg = EvaluationBOPConfig() 
# eval_cfg.nb_imgs_load = 100
eval_cfg.ds_name = 'ycbv'

print_mem_usg()
reader = BOPDataReader(BOP_DS_DIRS[eval_cfg.ds_name])
reader.load_scene(48, nb_img_loaded=eval_cfg.nb_imgs_load)

# accepted_objs = {
#     'obj_000010',
#     'obj_000016',
# }
accepted_objs = 'all'
tracker = Tracker(reader.get_intrinsics(), OBJECT_MODEL_DIRS[eval_cfg.ds_name], accepted_objs, eval_cfg.tracker_cfg)
tracker.init()

# localizer = Localizer('ycbv', THRESHOLD_DETECTOR)
localizer = Localizer('ycbv', eval_cfg.localizer_cfg, n_workers=2)

print('torch warming up...')
rgb = reader.get_img(0)
poses = localizer.predict(rgb, reader.K, n_coarse=1, n_refiner=1)
print('... torch warmed up!')

N_first = len(reader.img_lst)
# N_img = 50

LOCALIZE_EVERY = N_first # Never again
# LOCALIZE_EVERY = 10

for i in range(N_first):
    idx, rgb = reader.get_next_img()

    if i % LOCALIZE_EVERY == 0:
        poses = localizer.predict(rgb, reader.K, n_coarse=1, n_refiner=1)

    tracker.detected_bodies(poses)
    tracker.set_image(rgb)
    tracker.track()
    t = time.time()
    tracker.update_viewers()
    print('update_viewers (ms)', 1000*(time.time() - t))

    # cv2.waitKey(1)

