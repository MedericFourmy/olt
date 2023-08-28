import copy
import cv2
from collections import deque
from dataclasses import dataclass, field
import numpy as np
from thespian.actors import *
import json
import time
from datetime import timedelta
from PIL import Image
import logging

from olt.config import OBJ_MODEL_DIRS, MEGAPOSE_DATA_DIR, TrackerConfig, LocalizerConfig
from pyicg import Body




@dataclass
class TrackerRequest(object):
    img: np.ndarray = None
    img_id: int = -1
    img_time: float = -1.0
    poses_cosy: dict = field(default_factory=dict)
    poses_tracker: dict = field(default_factory=dict)
    poses_result: dict = field(default_factory=dict)
    cosy_base_id:int = -1
    result_log_time: float = -1.0
    send_all_newer_images: bool = False


    def has_image(self):
        return self.img is not None

    def has_poses_tracker(self):
        return len(self.poses_tracker.keys()) > 0
    
    def has_poses_result(self):
        return len(self.poses_result.keys()) > 0
    
    def has_poses_cosy(self):
        return len(self.poses_cosy.keys()) > 0
    
    def has_id(self):
        return self.img_id >= 0
    
    def has_cosy_base_id(self):
        return self.cosy_base_id >= 0
    
    # def draw(self, K):
    #     def get_base_frame_points():
    #         pts = np.zeros((4,3))
    #         pts[1,0] = 1
    #         pts[2,1] = 1
    #         pts[3,2] = 1
    #         return pts
        
    #     cv2.projectPoints


    
    @staticmethod
    def _get_sample_img_msg(img_nmb:int = 1):
        rgb_full_path = MEGAPOSE_DATA_DIR / f'bop_datasets/ycbv/test/000048/rgb/{img_nmb:06}.png'
        im = Image.open(rgb_full_path)
        rgb = np.array(im, dtype=np.uint8)

        return TrackerRequest(rgb)

class ImageBuffer(Actor):
    def __init__(self, *args, **kwargs):
        self.capacity = 60
        self.images = {} # image id to image
        self.image_id_deque = deque(maxlen=self.capacity)
        
        
        
        super().__init__(*args, **kwargs)

    def add_img(self, img, img_id):
        if len(self.image_id_deque) >= self.capacity:
            del self.images[self.image_id_deque.popleft()]
        self.images[img_id] = img
        self.image_id_deque.append(img_id)

    def get_image(self, img_id):
        if img_id not in self.images.keys():
            return None
        return self.images[img_id]
    
    def get_id(self):
        """
        checks for the highes current id and returns an increment
        """
        try:
            highest_id = self.image_id_deque[-1]
        except IndexError as e:
            highest_id = 0
        return highest_id + 1
    
    def receiveMessage(self, message, sender):
        if isinstance(message, TrackerRequest):
            logging.info("received an image to buffer")
            # save image to buffer
            if message.has_image() and message.has_id():
                self.add_img(message.img, message.img_id)
                self.send(sender, message)
            if message.has_image() and not message.has_id():
                new_id = self.get_id()
                self.add_img(message.img, new_id)
                message.img_id = new_id
                self.send(sender, message)

            # get image from buffer
            if message.has_id and not message.has_image():
                if message.send_all_newer_images:
                    new_msg = copy.deepcopy(message)
                    new_msg.send_all_newer_images = False

                    if new_msg.img_id + 1 > self.image_id_deque[-1]: 
                        # no newer images available
                        return
                    id_list = list(self.image_id_deque)
                    start_id = id_list.index(new_msg.img_id + 1)
                    all_ids_to_send = id_list[start_id:]
                    for _id in all_ids_to_send:
                        new_msg = copy.deepcopy(new_msg)
                        new_msg.img_id = _id
                        new_msg.img = self.get_image(_id)
                        self.send(sender, new_msg)
                else:
                    message.img = self.get_image(message.img_id)
                    self.send(sender, message)

        if isinstance(message, str) and message == "stats":
            s = f"buffer length {len(self.images.keys())}"
            self.send(sender, s)

        if isinstance(message, str) and message == "latest_image":
            try:
                highest_id = self.image_id_deque[-1]
                new_msg = TrackerRequest(self.images[highest_id], highest_id)
                self.send(sender, new_msg)
            except IndexError as e:
                pass
            
        
class ResultLoggerActor(Actor):
    def __init__(self, *args, **kwargs):
        self.store = {}
        # self.latest_result = (-1, -1) # (img_id, cosy_base_id)
        self.latest_result = -1
        self.highest_cosy_base_id = -1
     
        super().__init__(*args, **kwargs)

    def receiveMessage(self, message, sender):
        if isinstance(message, TrackerRequest):
            assert message.has_cosy_base_id()
            if message.has_poses_result():
                message.result_log_time = time.time()
                self.latest_result = max([self.latest_result, message.img_id])
                self.highest_cosy_base_id = max([self.highest_cosy_base_id, message.cosy_base_id])
                if message.img_id not in self.store.keys():
                    self.store[message.img_id] = [message]
                else:
                    self.store[message.img_id].append(message)

            # provide tracker results from previous image ids as estimate 
            elif message.has_id() and message.has_cosy_base_id():
                try:
                    for msg in self.store[message.img_id - 1]:
                        assert isinstance(msg, TrackerRequest)
                        if msg.cosy_base_id == message.cosy_base_id:
                            message.poses_tracker = msg.poses_result
                            self.send(sender, message, 0.1)
                            break

                except KeyError as e:
                    print("result not yet available.")

        if isinstance(message, str) and message == "print":
            print(self.store.keys())
            self.send(sender, self.store)


@dataclass
class ActorConfig():
    addressbook: dict = field(default_factory=dict)

class DispatcherActor(Actor):
    def __init__(self, *args, **kwargs):


        self.localizer_free = True
        self.latest_result = None
        self.latest_result_id = -1

        self.latest_cosy_base_id = -1
             
        super().__init__(*args, **kwargs)


    def receiveMessage(self, message, sender):
        if isinstance(message, ActorConfig):
            self.buffer = message.addressbook["buffer"]
            self.localizer = message.addressbook["localizer"]
            self.tracker = message.addressbook["tracker"]
            self.result_logger = message.addressbook["result_logger"]
            return

        if isinstance(message, TrackerRequest):
            if message.has_poses_result():
                # send message to result logger
                if self.latest_result_id < message.img_id:
                    self.latest_result = copy.deepcopy(message)
                self.send(self.result_logger, message)


                return
            if message.has_poses_tracker():
                # send the message to the tracker for refinement
                self.send(self.tracker, message)
                return
            if message.has_poses_cosy():
                self.send(self.tracker, message)
                # make sure that following messages are based on this result
                # get also all images following this' message id from the buffer
                self.send(self.buffer, TrackerRequest(img_id=message.img_id, 
                                                      send_all_newer_images=True, 
                                                      cosy_base_id=message.cosy_base_id))
                self.latest_cosy_base_id = max([message.cosy_base_id, self.latest_cosy_base_id])
                return

            if message.has_image() and not message.has_id():
                # put message to buffer to get id (and buffer)
                self.send(self.buffer, message)
                return
            
            if message.has_image() and message.has_id() and message.has_cosy_base_id():
                # trigger adding estimate from previous results
                self.send(self.result_logger, message)
                return



            # if message.has_image() and message.has_id():
            #     # send new task tp localizer
            #     self.send(self.localizer, message, 0.1)


            # if message.has_image() and message.has_id() and message.has_poses_cosy() and not message.has_poses_result():
            #     # got a message from a localizer and we should request a new detection with the newest image. Therefore, we request the latest image from the buffer.
            #     self.send(self.buffer, 'latest_image')

            #     # send message for refinement to tracker
            #     self.send(self.tracker, message, 0.1)




class LocalizerActor(Actor):
    def __init__(self, *args, **kwargs):

        self.buffer = None

        from olt.config import OBJ_MODEL_DIRS, MEGAPOSE_DATA_DIR, TrackerConfig, LocalizerConfig
        from olt.localizer import Localizer

        DS_NAME = 'ycbv'
        SCENE_ID = 48
        VIEW_ID = 1

        lcfg = LocalizerConfig
        lcfg.n_workers = 2

        self.localizer = Localizer(DS_NAME, lcfg)


        

        rgb_full_path = MEGAPOSE_DATA_DIR / 'bop_datasets/ycbv/test/000048/rgb/000001.png'
        scene_cam_full_path = MEGAPOSE_DATA_DIR / 'bop_datasets/ycbv/test/000048/scene_camera.json'
        d_scene_camera = json.loads(scene_cam_full_path.read_text())
        K = d_scene_camera[str(VIEW_ID)]['cam_K']
        self.K = np.array(K).reshape((3,3))

        im = Image.open(rgb_full_path)
        rgb = np.array(im, dtype=np.uint8)
        height, width, _ = rgb.shape


        self.last_img_id = -1
       

        self.POLLING = True
     
        super().__init__(*args, **kwargs)

    def receiveMessage(self, message, sender):
        if isinstance(message, ActorConfig):
            self.buffer = message.addressbook["buffer"]
            self.dispatcher = message.addressbook["dispatcher"]
            # self.localizer = message.addressbook["localizer"]
            # self.tracker = message.addressbook["tracker"]
            # self.result_logger = message.addressbook["result_logger"]
            return
        if isinstance(message, TrackerRequest):
            
            if message.has_image():
                assert message.has_id()
                if message.img_id > self.last_img_id:
                    poses = self.localizer.predict(message.img, self.K, n_coarse=1, n_refiner=3)
                    self.last_img_id = message.img_id
                    message.poses_cosy = poses
                    message.cosy_base_id = message.img_id
                    self.send(self.dispatcher ,message)
                else:
                    # we polled the same image twice. In the test cases, where
                    # images come slow, we need to poll later or we are deadlocked.
                    self.wakeupAfter(timedelta(seconds=0.9), payload="poll")
                    return
                
                if self.POLLING:
                    self.send(self.buffer, "latest_image")
        if isinstance(message, str) and message == "poll":
            self.send(self.buffer, "latest_image")

class TrackerActor(Actor):
    def __init__(self, *args, **kwargs):
        from olt.config import OBJ_MODEL_DIRS, MEGAPOSE_DATA_DIR, TrackerConfig, LocalizerConfig
        from olt.tracker import Tracker
        from olt.utils import Kres2intrinsics, print_mem_usage

        DS_NAME = 'ycbv'
        SCENE_ID = 48
        VIEW_ID = 1

        rgb_full_path = MEGAPOSE_DATA_DIR / 'bop_datasets/ycbv/test/000048/rgb/000001.png'
        scene_cam_full_path = MEGAPOSE_DATA_DIR / 'bop_datasets/ycbv/test/000048/scene_camera.json'
        d_scene_camera = json.loads(scene_cam_full_path.read_text())
        K = d_scene_camera[str(VIEW_ID)]['cam_K']
        K = np.array(K).reshape((3,3))

        im = Image.open(rgb_full_path)
        rgb = np.array(im, dtype=np.uint8)
        height, width, _ = rgb.shape
        intrinsics = Kres2intrinsics(K, width, height)

        accepted_objs = 'all'
        tcfg = TrackerConfig()
        tcfg.n_corr_iterations
        tcfg.viewer_display = False
        tcfg.viewer_save = True
        self.tracker = Tracker(intrinsics, OBJ_MODEL_DIRS[DS_NAME], accepted_objs, tcfg)
        self.tracker.init()
        super().__init__(*args, **kwargs)

    def receiveMessage(self, message, sender):
        if isinstance(message, TrackerRequest):
            if message.has_poses_cosy():
                self.tracker.detected_bodies(message.poses_cosy)
            elif message.has_poses_tracker():
                self.tracker.detected_bodies(message.poses_tracker)
            if message.has_image():
                self.tracker.set_image(message.img)
                self.tracker.track()
                message.poses_result = self.tracker.bodies
                message.poses_result = {}
                for name, body in self.tracker.bodies.items():
                    assert isinstance(body, Body)
                    message.poses_result[name] = body.world2body_pose
                self.tracker.update_viewers()
                self.send(sender, message)




if __name__ == "__main__":

    system = ActorSystem('multiprocQueueBase')
        


    localizer = system.createActor(LocalizerActor)
    tracker_1 = system.createActor(TrackerActor)

    msg = TrackerRequest._get_sample_img_msg()
    msg = system.ask(localizer, msg, 1)
    res = system.ask(tracker_1, msg, 1)

    print(res)

    system.shutdown()
    print("shutdown complete.")
