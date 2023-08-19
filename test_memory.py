# Inspiration: https://chase-seibert.github.io/blog/2013/08/03/diagnosing-memory-leaks-python.html

# Nothing -> 13 Mb
from olt.tracker import Tracker  # 93 Mb
from olt.config import OBJ_MODEL_DIRS, TrackerConfig
from olt.utils import get_mem_usage, print_mem_usage
print_mem_usage()
from olt.localizer import Localizer  # 530 Mb
import time

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
t1 = time.time()
tracker = Tracker(dummy_intrinsics, OBJ_MODEL_DIRS[DS_NAME], accepted_objs, tcfg)
print(f'tracker1 creation took {1000*(time.time() - t1)} ms')
tracker.init()
print(f'tracker1 creation+init took {1000*(time.time() - t1)} ms')
print_mem_usage()
