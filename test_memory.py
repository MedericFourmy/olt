# Inspiration: https://chase-seibert.github.io/blog/2013/08/03/diagnosing-memory-leaks-python.html

# Nothing -> 13 Mb
from olt.tracker import Tracker  # 93 Mb
from olt.config import BOP_DS_DIRS
# from localizer import Localizer  # 530 Mb

CAMERA_PATH = 'cam_d435_640.yaml'

DS_NAME = 'ycbv'
# Tracker creation without any object = 157 - 106 = 51 Mb
# accepted_objs = {
#     'obj_000010',
#     # 'obj_000016',
# }
accepted_objs = 'all'
tracker = Tracker(CAMERA_PATH, BOP_DS_DIRS[DS_NAME], 'tmp', accepted_objs)
tracker.init()

import resource
print(f'Memory usage: {int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000} (Mb)')