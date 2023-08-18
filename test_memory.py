# Inspiration: https://chase-seibert.github.io/blog/2013/08/03/diagnosing-memory-leaks-python.html

# Nothing -> 13 Mb
from olt.tracker import Tracker  # 93 Mb
from olt.config import BOP_DS_DIRS, TrackerConfig
from olt.utils import print_mem_usage
print_mem_usage()
from olt.localizer import Localizer  # 530 Mb

CAMERA_PATH = 'cam_d435_640.yaml'

DS_NAME = 'ycbv'
# Tracker creation without any object = 157 - 106 = 51 Mb
# accepted_objs = {
#     'obj_000010',
#     # 'obj_000016',
# }
accepted_objs = 'all'
tcfg = TrackerConfig()

dummy_intrinsics = {'fu': 1, 'fv': 1, 'ppu': 0, 'ppv': 0, 'width': 100, 'height': 100}
tracker = Tracker(dummy_intrinsics, BOP_DS_DIRS[DS_NAME], accepted_objs, tcfg)
tracker.init()

print_mem_usage()
