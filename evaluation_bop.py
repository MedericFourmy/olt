

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
from olt.utils import Kres2intrinsics
from olt.config import BOP_DS_DIRS, OBJECT_MODEL_DIRS

DS_NAME = 'ycbv'

THRESHOLD_DETECTOR = .9
NB_IMGS_LOAD = 100


print_mem_usg()
reader = BOPDataReader(BOP_DS_DIRS[DS_NAME])
reader.load_scene(48, nb_img_loaded=NB_IMGS_LOAD)

# accepted_objs = {
#     'obj_000010',
#     'obj_000016',
# }
accepted_objs = 'all'
tracker = Tracker(reader.get_intrinsics(), OBJECT_MODEL_DIRS[DS_NAME], 'tmp', accepted_objs)
tracker.init()

localizer = Localizer('ycbv', THRESHOLD_DETECTOR)

print('torch warming up...')
rgb = reader.get_img(0)
poses = localizer.predict(rgb, reader.K, n_coarse=1, n_refiner=3)
print('... torch warmed up!')

N_img = len(reader.img_lst)
# N_img = 10

# for i in range(N_img):
#     idx, rgb = reader.get_next_img()

#     # update K where we can
#     # tracker.update_K(K)
#     # if i == 0:
#     #     poses = localizer.predict(rgb, reader.K, n_coarse=1, n_refiner=1)

#     tracker.detected_bodies(poses)
#     tracker.set_image(rgb)
#     tracker.track()
#     t = time.time()
#     tracker.update_viewers()
#     print('update_viewers (ms)', 1000*(time.time() - t))

#     cv2.waitKey(1)

