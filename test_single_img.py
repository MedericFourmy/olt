import cv2
import time
import resource
print_mem_usage = lambda: print(f'Memory usage: {int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000} (Mb)')

from olt.tracker import Tracker
from olt.localizer import Localizer
from olt.evaluation_tools import BOPDataReader
from olt.config import BOP_DS_DIRS, OBJECT_MODEL_DIRS

DS_NAME = 'ycbv'
reader = BOPDataReader(BOP_DS_DIRS[DS_NAME])
reader.load_scene(48, nb_img_loaded=1)
print_mem_usage()

## HAPPYPOSE
THRESHOLD_DETECTOR = 0.6

# accepted_objs = {
#     'obj_000010',
#     'obj_000016',
# }
accepted_objs = 'all'
tracker = Tracker(reader.get_intrinsics(), OBJECT_MODEL_DIRS[DS_NAME], 'tmp', accepted_objs)
print_mem_usage()

tracker.init()
print_mem_usage()

localizer = Localizer('ycbv', THRESHOLD_DETECTOR, n_workers=2)
print_mem_usage()

rgb = reader.get_img(0)
poses = localizer.predict(rgb, reader.K, n_coarse=1, n_refiner=1)
print_mem_usage()

poses = localizer.predict(rgb, reader.K, n_coarse=1, n_refiner=1)
print_mem_usage()


tracker.detected_bodies(poses)
tracker.set_image(rgb)
tracker.track()
t = time.time()
tracker.update_viewers()
print('update_viewers (ms)', 1000*(time.time() - t))
print_mem_usage()

# cv2.waitKey(0)

