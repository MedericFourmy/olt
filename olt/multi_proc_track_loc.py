import time
import logging
import multiprocessing as mp

from collections import deque

import cv2
import numpy as np
import json
import time
from PIL import Image

from olt.tracker import Tracker

from olt.utils import Kres2intrinsics, print_mem_usage
from olt.config import OBJ_MODEL_DIRS, HAPPYPOSE_DATA_DIR, TrackerConfig, LocalizerConfig

from queue import Empty, Full


logger = logging.Logger("Main")

#############################
"""
    Image + info loading

    For this example, manually get one image + cam intrinsics of the views from a bop dataset
    To load a full scene, use BOPDatasetReader in evaluation_tools.py
"""
DS_NAME = 'ycbv'
SCENE_ID = 48
VIEW_ID = 1

scene_id_str = '{SCENE_ID:06}'
view_id_str = '{VIEW_ID:06}'
rgb_full_path = HAPPYPOSE_DATA_DIR / 'bop_datasets/ycbv/test/000048/rgb/000001.png'
scene_cam_full_path = HAPPYPOSE_DATA_DIR / 'bop_datasets/ycbv/test/000048/scene_camera.json'
d_scene_camera = json.loads(scene_cam_full_path.read_text())
K = d_scene_camera[str(VIEW_ID)]['cam_K']
K = np.array(K).reshape((3,3))

im = Image.open(rgb_full_path)
rgb = np.array(im, dtype=np.uint8)
height, width, _ = rgb.shape
intrinsics = Kres2intrinsics(K, width, height)
#############################

class TrackerProcess(object):
    def __init__(self) -> None:
        self.q_manager = mp.Manager()

        self.base_img_id = -1
        self.last_image_id = -1

        self.in_q = self.q_manager.Queue(maxsize=60)
        self.poses_q = self.q_manager.Queue(maxsize=1)
        self.out_q = self.q_manager.Queue(maxsize=60)
        self.stop_event = self.q_manager.Event()
        
        self.proc = mp.Process(target=self.run, args=(self.stop_event, self.poses_q, self.in_q, self.out_q, ))

    def start(self):
        self.proc.start()

    def add_pose_init(self, img_id, poses, img_buffer=None):
        self.base_img_id = img_id
        self.last_image_id = img_id - 1
        self.poses_q.put((img_id, poses))

    def add_image(self, img_id, img_time, img):
        if img_id > self.last_image_id:
            self.in_q.put((img_id, img_time, img))
                




    @staticmethod
    def run(stop_event, poses_q, img_q, out_q):
        accepted_objs = 'all'
        tcfg = TrackerConfig()
        tcfg.n_corr_iterations
        tcfg.viewer_display = False
        tcfg.viewer_save = True
        tracker = Tracker(intrinsics, OBJ_MODEL_DIRS[DS_NAME], accepted_objs, tcfg)
        tracker.init()

        last_img_id = -1

        while not stop_event.is_set():
            try:
                # print(image_q.qsize())
                (img_id, img_time, rgb, init_poses) = poses_q.get_nowait()
                tracker.detected_bodies(init_poses)
                tracker.set_image(rgb)
                tracker.track()
                last_img_id = img_id

                out_q.put((img_id, img_time, time.time(), tracker.bodies))
            except Empty as e:
                pass
            try:
                # print(image_q.qsize())
                (img_id, img_time, rgb) = img_q.get_nowait()
                if img_id <= last_img_id:
                    continue
                tracker.set_image(rgb)
                tracker.track()

                out_q.put((img_id, img_time, time.time(), tracker.bodies))

            except Empty as e:
                pass

        logger.warn(f"shutting down tracker worker")


class MultiProcessTracker(object):
    def __init__(self) -> None:
        self.q_manager = mp.Manager()
        self.q_dict = {}
        self.image_buffer = deque(maxlen=60)
        self.img_id = -1

    

        self.q_dict["loc_in_q"] = self.q_manager.Queue(maxsize=1)
        self.q_dict["loc_out_q"] = self.q_manager.Queue(maxsize=10)
        self.q_dict["loc_stop_event"] = self.q_manager.Event()
        self.loc_proc = mp.Process(target=MultiProcessTracker.localizer_proc, args=(self.q_dict["loc_stop_event"], self.q_dict["loc_in_q"], self.q_dict["loc_out_q"], ))
        

        self.trackerprocess = TrackerProcess()

        # self.q_dict["trackproc_1_in_q"] = self.q_manager.Queue(maxsize=60)
        # self.q_dict["trackproc_1_poses_q"] = self.q_manager.Queue(maxsize=1)
        # self.q_dict["trackproc_1_out_q"] = self.q_manager.Queue(maxsize=60)
        # self.q_dict["trackproc_1_stop_event"] = self.q_manager.Event()
        # self.trackproc_1 = mp.Process(target=MultiProcessTracker.tracker_proc, args=(self.q_dict["trackproc_1_stop_event"], self.q_dict["trackproc_1_poses_q"], self.q_dict["trackproc_1_in_q"], self.q_dict["trackproc_1_out_q"], ))

        # self.q_dict["trackproc_2_in_q"] = self.q_manager.Queue(maxsize=60)
        # self.q_dict["trackproc_2_poses_q"] = self.q_manager.Queue(maxsize=1)
        # self.q_dict["trackproc_2_out_q"] = self.q_manager.Queue(maxsize=60)
        # self.q_dict["trackproc_2_stop_event"] = self.q_manager.Event()
        # self.trackproc_2 = mp.Process(target=MultiProcessTracker.tracker_proc, args=(self.q_dict["trackproc_2_stop_event"], self.q_dict["trackproc_2_poses_q"], self.q_dict["trackproc_2_in_q"], self.q_dict["trackproc_2_out_q"], ))


        self.loc_proc.start()
        self.trackerprocess.start()



    def run(self):
        while True:

            # get new image
            try:
                (img_id, img_time, rgb, k) = self.in_q.get_nowait()
                # feed cosypose
                self.q_dict["loc_in_q"].put((img_id, img_time, rgb, k))
                # feed image to trackers
                self.q_dict["trackproc_1_in_q"].put((img_id, img_time, rgb))
                self.q_dict["trackproc_2_in_q"].put((img_id, img_time, rgb))

            except Empty as e:
                pass

            # get cosypose results
            try:
                (img_id, img_time, res_time, poses) = self.q_dict["loc_out_q"].get_nowait()
                
                # feed result to the tracker based on the oldest cosypose initalization
                self.q_dict["trackproc_1_in_q"].put((img_id, img_time, rgb))
                self.q_dict["trackproc_2_in_q"].put((img_id, img_time, rgb))

            except Empty as e:
                pass
        




    def track(self, img, img_time, k):
        self.img_id += 1
        self.image_buffer.append((self.img_id, img_time, img, k))
        print('in track')

        
        


    @staticmethod
    def tracker_proc(stop_event, poses_q, img_q, out_q):
        
        accepted_objs = 'all'
        tcfg = TrackerConfig()
        tcfg.n_corr_iterations
        tcfg.viewer_display = False
        tcfg.viewer_save = True
        tracker = Tracker(intrinsics, OBJ_MODEL_DIRS[DS_NAME], accepted_objs, tcfg)
        tracker.init()

        last_img_id = -1

        while not stop_event.is_set():
            try:
                # print(image_q.qsize())
                (img_id, img_time, rgb, init_poses) = poses_q.get_nowait()
                tracker.detected_bodies(init_poses)
                tracker.set_image(rgb)
                tracker.track()
                last_img_id = img_id

                out_q.put((img_id, img_time, time.time(), tracker.bodies))
            except Empty as e:
                pass
            try:
                # print(image_q.qsize())
                (img_id, img_time, rgb) = img_q.get_nowait()
                if img_id <= last_img_id:
                    continue
                tracker.set_image(rgb)
                tracker.track()

                out_q.put((img_id, img_time, time.time(), tracker.bodies))

            except Empty as e:
                pass

            logger.warning(f"shutting down tracker worker")



    @staticmethod
    def localizer_proc(stop_event, in_q, out_q):

        from olt.localizer import Localizer

        lcfg = LocalizerConfig
        lcfg.n_workers = 2
        localizer = Localizer(DS_NAME, lcfg)

        while not stop_event.is_set():
            try:
                (img_id, img_time, rgb, K) = in_q.get_nowait()
            except Empty as e:
                continue

            ### do tracking ###
            poses, scores = localizer.predict(rgb, K, n_coarse=1, n_refiner=3)

            res_time = time.time()

            out_q.put((img_id, img_time, res_time, poses))
            logger.info(f"CosyPose prediction: \n{img_id}, {img_time}, {res_time}, {poses}")


        print(f"shutting down worker")

        






def main():
    # mp.set_start_method("fork")
    mptrack = MultiProcessTracker()

    # load image sequence
    imgs = []

    # for time, img in imgs:
    mptrack.track(img_time=time.time(), img=rgb, k=K)

    mptrack.q_dict["loc_stop_event"].set()

    mptrack.trackerprocess.stop_event.set()




if __name__ == "__main__":
    main()