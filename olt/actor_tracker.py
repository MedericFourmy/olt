import copy
import os
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

from olt.config import OBJ_MODEL_DIRS, MEGAPOSE_DATA_DIR, TrackerConfig, LocalizerConfig, logcfg
from pym3t import Body

import functools
import cProfile, pstats

DEBUG = False

@dataclass
class ProcessStats(object):
    memory: int = -1
    load: float = -1.0
    # profile_stats: 


def get_name(cl, dt):
    dt = int(dt*1000000) # to ns
    class_str = str(cl).split(".")[-1].removesuffix("'>")
    s = f"{dt:010d}_{class_str}.profile"
    return s


def measure_load(func):
    @functools.wraps(func)
    def wrapper_load_measure(*args, **kwargs):
        if DEBUG:
            self = args[0]
            start_time = time.perf_counter()
            with cProfile.Profile() as pr:
                value = func(*args, **kwargs)

                end_time = time.perf_counter()
                run_time = end_time - start_time
                pr.dump_stats(get_name(self, run_time))

                
                
                # pr.print_stats("cumulative")
                # pr.create_stats()
                # logging.info(pr.stats)
                # pr.print_stats()
            

            try:
                self.start_times.append(start_time)
            except AttributeError as e:
                self.start_times = deque([start_time], maxlen=10)
            try:
                self.end_times.append(end_time)
            except AttributeError as e:
                self.end_times = deque([end_time], maxlen=10)

            load = (np.sum(np.array(self.end_times)- np.array(self.start_times))) / (self.end_times[-1] - self.start_times[0])
            if run_time > 0.03:
                logging.warn(f"{type(args[0])} was running with {100* load} % and the las call took {run_time} s.")
            else:
                logging.info(f"{type(args[0])} was running with {100* load} % and the las call took {run_time} s.")
        else:
            value = func(*args, **kwargs)

        return value
    return wrapper_load_measure

@dataclass
class TrackerRequest(object):
    img: np.ndarray = None
    depth: np.ndarray = None
    img_id: int = -1
    img_time: float = -1.0
    poses_cosy: dict = field(default_factory=dict)
    poses_tracker: dict = field(default_factory=dict)
    poses_result: dict = field(default_factory=dict)
    cosy_base_id:int = -1
    result_log_time: float = -1.0
    send_all_newer_images: bool = False
    timeout_policy: str = 'newest'
    timeout: float = 0.5

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, self.__class__):
            return False
        
        for attr1, attr2 in zip(self.__dir__(), __value.__dir__()):
            if attr1 != attr2:
                return False
           
        if any(self.__dict__ != __value.__dict__):
            return False
        
        return True


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

class ImageStreamerActor(Actor):
    def __init__(self, *args, **kwargs):
        self.active = False
        self.index = 1
        self.receivers = {}
        self.dt = 0.5
        self.start_time = time.time()
        super().__init__(*args, **kwargs)

    def receiveMessage(self, message, sender):
        if isinstance(message, ActorConfig):
            self.receivers.update(message.addressbook)
        if isinstance(message, str) and message == "start":
            self.active = True
            self.wakeupAfter(timedelta(seconds=self.dt))
        if isinstance(message, str) and message == "stop":
            self.active = False
        if isinstance(message, str) and "hz" in message: # "hz 30"
            hz = float(message.strip("hz "))
            self.dt = 1.0/hz
        if isinstance(message, str) and message in self.__dir__():
            self.send(sender, self.__getattribute__(message))

        
        if isinstance(message, WakeupMessage) and self.active:
            self.wakeupAfter(timedelta(seconds=self.dt))
            try:
                new_msg = TrackerRequest._get_sample_img_msg(self.index)
            except FileNotFoundError as e:
                self.active = False
                logging.warn("I streamed the whole dataset of images and exit now")
                self.send(self.myAddress, ActorExitRequest())
                return
            
            new_msg.img_time = time.time()
            for target, addr in self.receivers.items():
                logging.info(f"send image {self.index} to {target} at {new_msg.img_time}")
                self.send(addr, new_msg)
            self.index += 1

            


class ImageBuffer(Actor):
    def __init__(self, *args, **kwargs):
        self.capacity = 120
        self.images = {} # image id to image
        self.image_id_deque = deque(maxlen=self.capacity)

        self.open_result_requests = {} # img_id to message waiting for that image_id
        
        
        
        super().__init__(*args, **kwargs)

    def serve_message(self, img_id):
        for sender, message in self.open_result_requests[img_id]: 
            assert isinstance(message, TrackerRequest)
            self.send(sender, self.images[message.img_id])

        logging.info(f"Buffer served {len(self.open_result_requests[img_id])} requests for img_id {img_id}.")
        del self.open_result_requests[img_id]

    def add_result_request(self, img_id, sender, message):
        sender_msg_tuple = (sender, message)
        if img_id not in self.open_result_requests.keys():
            self.open_result_requests[img_id] = [sender_msg_tuple]
        else:
            self.open_result_requests[img_id].append(sender_msg_tuple)

    def add_img(self, img, img_id):
        if len(self.image_id_deque) >= self.capacity:
            del self.images[self.image_id_deque.popleft()]
        self.images[img_id] = img
        self.image_id_deque.append(img_id)

    def get_image(self, img_id):
        if img_id not in self.images.keys():
            raise IndexError(f"image {img_id} is not in buffer.")
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
    
    @measure_load
    def receiveMessage(self, message, sender):
        if isinstance(message, TrackerRequest):
            # save image to buffer
            if message.has_image() and message.has_id():
                logging.info(f"received image with given id {message.img_id} to buffer")
                self.add_img(message, message.img_id)
                if message.img_id in self.open_result_requests.keys():
                    self.serve_message(message.img_id)
                # self.send(sender, message)
            if message.has_image() and not message.has_id():
                new_id = self.get_id()
                logging.info(f"received image {new_id} to buffer.")
                self.add_img(message, new_id)
                message.img_id = new_id
                if message.img_id in self.open_result_requests.keys():
                    self.serve_message(message.img_id)

                # self.send(sender, message)

            # get image from buffer
            if message.has_id and not message.has_image():
                if message.send_all_newer_images:
                    new_msg = copy.deepcopy(message)
                    new_msg.send_all_newer_images = False

                    if new_msg.img_id + 1 > self.image_id_deque[-1]: 
                        logging.error(f"the requested image id {new_msg.img_id + 1} is newer than all buffer images.")
                        return
                    if new_msg.img_id < self.image_id_deque[0]:
                        logging.warn(f"the requested image id {new_msg.img_id} is not in the buffer anymore.")
                        return
                    
                    id_list = list(self.image_id_deque)
                    start_id = id_list.index(new_msg.img_id + 1)
                    all_ids_to_send = id_list[start_id:]
                    for _id in all_ids_to_send:
                        # new_msg = copy.deepcopy(new_msg)
                        # new_msg.img_id = _id
                        new_msg = self.get_image(_id)
                        assert isinstance(new_msg, TrackerRequest)
                        new_msg.cosy_base_id = message.cosy_base_id
                        self.send(sender, new_msg)
                else:
                    logging.info(f"Buffer received image request for img {message.img_id}.")
                    try:
                        message = self.get_image(message.img_id)
                        logging.info(f"Buffer serves image request {message.img_id} directly.")
                        self.send(sender, message)
                    except IndexError as e:
                        logging.info(f"Buffer will answer request for img {message.img_id} ASAP.")
                        self.add_result_request(img_id=message.img_id, sender=sender, message=message)

        if isinstance(message, str) and message == "stats":
            s = f"buffer length {len(self.images.keys())}"
            self.send(sender, s)

        if isinstance(message, str) and message == "latest_image":
            try:
                highest_id = self.image_id_deque[-1]
                new_msg = self.get_image(highest_id)
                # new_msg = TrackerRequest(self.images[highest_id], highest_id)
                self.send(sender, new_msg)
            except IndexError as e:
                pass
        if isinstance(message, str) and message == "latest_image_id":
            try:
                highest_id = self.image_id_deque[-1]
                assert isinstance(highest_id, int)
                self.send(sender, highest_id)
            except IndexError as e:
                self.send(sender, -1)
            
        
class ResultLoggerActor(Actor):
    def __init__(self, *args, **kwargs):
        self.store = {}
        self.img_ids = deque(maxlen=20)
        # self.latest_result = (-1, -1) # (img_id, cosy_base_id)
        self.latest_result = -1
        self.earliest_result = 10000
        self.highest_cosy_base_id = -1
        self.open_result_requests = {} # img_id to message waiting for that image_id

        self.message_triggers = []

        self.registered_addresses = []
     
        super().__init__(*args, **kwargs)

    def serve_message(self, img_id):
        for sender, message in self.open_result_requests[img_id]: 
            assert isinstance(message, TrackerRequest)

            # serve other result requests without cosy id
            if not message.has_cosy_base_id():
                res = self.store[message.img_id][-1]
                self.send(sender, res)

            if message.has_cosy_base_id():
                try:
                    for msg in self.store[img_id]:
                        assert isinstance(msg, TrackerRequest)
                        if msg.cosy_base_id == message.cosy_base_id:
                            message.poses_tracker = msg.poses_result
                            self.send(sender, message)
                            break
                
                except KeyError as e:
                    assert False
                # logging.error(f"result {message.img_id - 1} not yet available for msg {message.img_id}.")
            
        # removing the served request
        logging.info(f"Served all requests for {message.img_id}.")
        del self.open_result_requests[img_id] 

    def add_result_request(self, img_id, sender, message):
        assert isinstance(message, TrackerRequest)
        sender_msg_tuple = (sender, message)
        if img_id not in self.open_result_requests.keys():
            self.open_result_requests[img_id] = [sender_msg_tuple]
        else:
            self.open_result_requests[img_id].append(sender_msg_tuple)

        if message.timeout_policy == "newest":
            def trigger(msg: TrackerRequest):
                i = 0
                while i < 2:
                    yield [msg.img_id > message.img_id, sender_msg_tuple][i]
                    i += 1
            self.message_triggers.append(trigger)

    def served(self, sender_msg_tuple):
        # remove the request from the open req dict
        sender, msg = sender_msg_tuple
        assert isinstance(msg, TrackerRequest)
        self.open_result_requests[msg.img_id].remove(sender_msg_tuple)
    
    @measure_load
    def receiveMessage(self, message, sender):
        if isinstance(message, TrackerRequest):
            # assert message.has_cosy_base_id()
            if message.has_poses_result():

                for receiver in self.registered_addresses:
                    self.send(receiver, message)

                message.result_log_time = time.time()
                self.earliest_result = min([self.earliest_result, message.img_id])
                self.latest_result = max([self.latest_result, message.img_id])
                self.highest_cosy_base_id = max([self.highest_cosy_base_id, message.cosy_base_id])

                if message.img_id not in self.store.keys():
                    if self.img_ids.__len__() >= self.img_ids.maxlen:
                        # keep the stored results small
                        to_delete_key = self.img_ids.popleft()
                        del self.store[to_delete_key]
                    self.img_ids.append(message.img_id)

                    self.store[message.img_id] = [message]
                else:
                    self.store[message.img_id].append(message)

                if message.img_id in self.open_result_requests.keys():
                    
                    logging.error(f"Got message {message.img_id} that a message requested.")
                    self.serve_message(message.img_id)
                
                # if a trigger is true for a message, it will first yield true and then the orig sender and message
                for trigger in self.message_triggers:
                    it = trigger(message)
                    if it.__next__():
                        orig_sender, orig_msg = it.__next__()
                        self.send(orig_sender, message)
                        try:
                            self.served((orig_sender, orig_msg))
                        except ValueError as e:
                            pass

            # provide tracker results from previous image ids as estimate 
            elif message.has_id() and message.has_cosy_base_id():
                try:
                    for msg in self.store[message.img_id - 1]:
                        assert isinstance(msg, TrackerRequest)
                        if msg.cosy_base_id == message.cosy_base_id:
                            message.poses_tracker = msg.poses_result
                            self.send(sender, message)
                            break

                except KeyError as e:
                    logging.error(f"result {message.img_id - 1} not yet available for msg {message.img_id}.")
                    # self.open_result_requests[message.img_id - 1] = (sender, message)
                    self.add_result_request(message.img_id - 1, sender, message)
            # provide result for any img_id
            elif message.has_id() and not message.has_cosy_base_id():

                
                try:
                    if self.earliest_result > message.img_id:
                        logging.error(f"result {message.img_id} not available. Requested img_id smaller than all results. provide earliest result instead.")
                        self.send(sender, self.store[self.earliest_result][-1])
                        return
                except KeyError as e:
                    logging.error(f"result {message.img_id} not available. No results available. provide empty result instead.")
                    self.send(sender, TrackerRequest(img_id=message.img_id))
                    return
                try: 
                    
                    res = self.store[message.img_id][-1]
                    self.send(sender, res)
                    return
                except KeyError as e:
                    logging.error(f"result {message.img_id} not yet available. Will answer ASAP")
                    # self.open_result_requests[message.img_id] = (sender, message)
                    self.add_result_request(message.img_id, sender, message)
                    self.wakeupAfter(message.timeout, (sender, message))
                    return
                
        if isinstance(message, WakeupMessage):
            orig_sender, orig_msg = message.payload
            assert isinstance(orig_msg, TrackerRequest)
            if orig_msg.timeout_policy == "none":
                if orig_msg.img_id in self.open_result_requests.keys():
                    if (orig_sender, orig_msg) in self.open_result_requests[orig_msg.img_id]:
                        # the message is still there, so we must send None
                        logging.warn(f"request for result {orig_msg.img_id} could not be ansered in time so we sende None.")
                        self.send(orig_sender, None)

                        self.open_result_requests[orig_msg.img_id].remove((orig_sender, orig_msg))
                return
            
            if orig_msg.timeout_policy == "newest":
                if orig_msg.img_id in self.open_result_requests.keys():
                    if (orig_sender, orig_msg) in self.open_result_requests[orig_msg.img_id]:
                        # the message is still there, so we must send the newest available result
                        logging.warn(f"request for result {orig_msg.img_id} could not be ansered in time so we sende the newest one.")
                        self.send(orig_sender, self.store[self.latest_result][-1])

                        
                        self.open_result_requests[orig_msg.img_id].remove((orig_sender, orig_msg))
                return
            
        if isinstance(message, str) and message == "register":
            self.registered_addresses.append(sender)

                
        
        if isinstance(message, str) and message == "latest_estimate":
            try:
                self.send(sender, self.store[self.latest_result][-1])
            except KeyError as e:
                self.send(sender, None)

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

    @measure_load
    def receiveMessage(self, message, sender):
        if isinstance(message, ActorConfig):
            self.buffer = message.addressbook["buffer"]
            self.localizer = message.addressbook["localizer"]
            self.tracker = message.addressbook["tracker"]
            self.result_logger = message.addressbook["result_logger"]
            return

        if isinstance(message, TrackerRequest):
            logging.debug(f"Dispatcher received message with img_id {message.img_id}, cosy_poses {message.has_poses_cosy}, tracker_poses {message.has_poses_tracker}, and result_poses {message.has_poses_result}")
            if message.has_poses_result():
                # send message to result logger
                if self.latest_result_id < message.img_id:
                    self.latest_result_id = message.img_id
                    # self.latest_result = copy.deepcopy(message)
                raise ValueError(f"The dispatcher should not get these messages: {message}")
                self.send(self.result_logger, message)
                return
            if message.has_poses_tracker():
                # send the message to the tracker for refinement
                raise ValueError(f"The dispatcher should not get these messages: {message}")
                self.send(self.tracker, (self.result_logger, message))
                return
            if message.has_poses_cosy():
                self.send(self.tracker, (self.result_logger, message))
                # make sure that following messages are based on this result
                # get also all images following this' message id from the buffer
                # self.send(self.buffer, TrackerRequest(img_id=message.img_id, 
                #                                       send_all_newer_images=True, 
                #                                       cosy_base_id=message.cosy_base_id))
                self.latest_cosy_base_id = max([message.cosy_base_id, self.latest_cosy_base_id])
                return

            if message.has_image() and not message.has_id():
                # put message to buffer to get id (and buffer)
                self.send(self.buffer, message)
                return
            
            if message.has_image() and message.has_id() and message.has_cosy_base_id():
                # trigger adding estimate from previous results
                raise ValueError(f"The dispatcher should not get these messages: {message}")
                self.send(self.result_logger, message)
                return
            
            logging.error(f"unhandled message: {message}")



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

        logging.info("Starting Localizer")

        self.buffer = None

        from olt.config import OBJ_MODEL_DIRS, MEGAPOSE_DATA_DIR, TrackerConfig, LocalizerConfig
        from olt.localizer import Localizer

        DS_NAME = 'ycbv'
        SCENE_ID = 48
        VIEW_ID = 1

        lcfg = LocalizerConfig
        # lcfg.n_workers = 2

        lcfg.renderer_name = 'bullet'  # higher AR, bit slower
        lcfg.training_type = 'synt+real'
        lcfg.n_workers = 0


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

    def predict(self, img):
        return self.localizer.predict(img, self.K, n_coarse=1, n_refiner=3)
    
    def exit(self):
        logging.warn("killing localizer from inside.")
        os.kill(os.getpid(), 9)

    @measure_load
    def receiveMessage(self, message, sender):
        if isinstance(message, ActorConfig):
            self.buffer = message.addressbook["buffer"]
            # self.dispatcher = message.addressbook["dispatcher"]
            self.trackers = message.addressbook["trackers"]
            # self.localizer = message.addressbook["localizer"]
            # self.tracker = message.addressbook["tracker"]
            # self.result_logger = message.addressbook["result_logger"]
            return
        if isinstance(message, TrackerRequest):
            
            if message.has_image():
                if not message.has_id():
                    # do prediction on image and return result, e.g. for warmup
                    logging.info("performing (warmup) prediction.")
                    poses, scores = self.predict(message.img)
                    # message.poses_cosy = poses
                    self.send(sender ,poses)
                    return

                assert message.has_id()
                if message.img_id > self.last_img_id:
                    logging.info(f"Starting prediction for img_id {message.img_id}.")
                    poses, scores = self.predict(message.img)
                    self.last_img_id = message.img_id
                    message.poses_cosy = poses
                    message.cosy_base_id = message.img_id
                    self.send(self.trackers ,message)
                else:
                    # we polled the same image twice. In the test cases, where
                    # images come slow, we need to poll later or we are deadlocked.
                    logging.info(f"we polled image {message.img_id} before. Trying again later.")
                    self.wakeupAfter(timedelta(seconds=0.1), payload="poll")
                    return
                
                if self.POLLING:
                    logging.info(f"Polling new img after finishing a prediction.")
                    self.send(self.buffer, "latest_image")
        if isinstance(message, WakeupMessage) and message.payload == "poll":
            logging.info(f"Polling new img from wakeup message.")
            self.send(self.buffer, "latest_image")

        if isinstance(message, str) and message == "poll":
            logging.info(f"Polling new img from outside trigger.")

            self.send(self.buffer, "latest_image")

        # if isinstance(message, ActorExitRequest) or isinstance(message, str) and message == "exit":
        #     self.send(sender, os.getpid())
            # self.exit()


class TrackerManager(Actor):
    def __init__(self, *args, **kwargs):

        self.trackers = []
            
        self.last_tracker = 0
        self.num_trackers = 3
     
        super().__init__(*args, **kwargs)

    def reinit_next_tracker(self, message):
        self.last_tracker += 1
        
        logging.info(f"sending tracker reset req with cosy_base_id {message.cosy_base_id} to tracker {self.last_tracker % self.num_trackers}")
        self.send(self.trackers[self.last_tracker % self.num_trackers], message)

    def add_tracker(self, cfg):
        self.trackers.append(self.createActor(TrackerActor))
        self.send(self.trackers[-1], cfg)


    def receiveMessage(self, message, sender):
        if isinstance(message, ActorConfig):
            self.cfg = message
            while len(self.trackers) < self.num_trackers:
                self.add_tracker(self.cfg)
            # for tracker in self.trackers:
            #     self.send(tracker, message)
            self.image_buffer = message.addressbook["buffer"]
            self.result_logger = message.addressbook["result_logger"]
        
        if isinstance(message, TrackerRequest):
            if message.has_image() and message.has_poses_cosy() and message.has_cosy_base_id():
                self.reinit_next_tracker(message)


class TrackerActor(Actor):
    def __init__(self, *args, **kwargs):
        from olt.config import OBJ_MODEL_DIRS, MEGAPOSE_DATA_DIR, TrackerConfig, LocalizerConfig
        from olt.tracker import Tracker
        from olt.utils import Kres2intrinsics, print_mem_usage

        self.STRIP_IMAGE = True
        self.POLLING = True
        self.ADVANCED_POLLING = False


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
        # self.tracker = Tracker(intrinsics, OBJ_MODEL_DIRS[DS_NAME], accepted_objs, tcfg)
        self.tracker = Tracker(OBJ_MODEL_DIRS[DS_NAME], accepted_objs, tcfg, rgb_intrinsics=intrinsics)

        self.tracker.init()


        self.cosy_base_id = -1
        self.last_img_id = -1

        super().__init__(*args, **kwargs)

    @measure_load
    def receiveMessage(self, message, sender):
        if isinstance(message, ActorConfig):
            self.image_buffer = message.addressbook["buffer"]
            self.result_logger = message.addressbook["result_logger"]
        if isinstance(message, tuple):
            sender, message = message
        if isinstance(message, TrackerRequest):
            logging.info(f"will perform tracking on image {message.img_id}.")
            if message.has_cosy_base_id():
                # Either this image was run through Localizer actor

                # tracker is on new track of images
                logging.info(f"Tracker is on cosy_base_id track {message.cosy_base_id}.")
                self.cosy_base_id = message.cosy_base_id
                assert message.img_id == message.cosy_base_id
                # assert message.img_id >= message.cosy_base_id
            elif self.last_img_id+1 != message.img_id:
                logging.info(f"Tracker ignores image {message.img_id}, because it is on another track ({self.cosy_base_id}).")
                return
            else:
                # if message came from ImageBuffer, it has no cosy id yet
                message.cosy_base_id = self.cosy_base_id
            if message.has_poses_cosy():
                self.tracker.detected_bodies(message.poses_cosy)

                if self.ADVANCED_POLLING:
                    self.send(self.image_buffer, TrackerRequest(img_id=message.img_id+1))
                    self.send(self.image_buffer, TrackerRequest(img_id=message.img_id+2))
                    self.send(self.image_buffer, TrackerRequest(img_id=message.img_id+3))

            elif message.has_poses_tracker():
                logging.error('!!!!!!!!!!!!!!!!!!!!!! message.has_poses_tracker() should not be True')
                # message.poses_tracker contains poses from the prev time
                # if images in order, the bodies need no update
                # update only bodies that are avialable
                self.tracker.detected_bodies(message.poses_tracker)
                
            if not message.has_poses_cosy() and not message.has_poses_tracker():
                # the message must be consecutive!
                assert message.img_id == self.last_img_id + 1

            if message.has_image():
                if self.POLLING:
                    logging.info(f"Tracker is polling for img {message.img_id+1} on cosy_base_id track {self.cosy_base_id}.")
                    
                    if self.ADVANCED_POLLING:
                        self.send(self.image_buffer, TrackerRequest(img_id=message.img_id+4))
                    else:
                        self.send(self.image_buffer, TrackerRequest(img_id=message.img_id+1))

                self.tracker.set_image(message.img)
                self.tracker.track()
                self.last_img_id = message.img_id

                # add tracking results to message
                message.poses_result = {}
                preds, active_tracks  = self.tracker.get_current_preds()
                for name, body_trans in preds.items():
                    if name in active_tracks:
                        assert body_trans.shape == (4,4)
                        message.poses_result[name] = body_trans

                logging.info(f"Tracking on image {message.img_id} complete.")
                if self.STRIP_IMAGE:
                    message.img = None
                self.send(self.result_logger, message)

                

            # logging.warn(f"unhandled message in tracker: {message}")

    # def receiveMessage(self, message, sender):
    #     if isinstance(message, tuple):
    #         sender, message = message
    #     if isinstance(message, TrackerRequest):
    #         logging.info(f"will perform tracking on image {message.img_id}.")
    #         if message.has_poses_cosy():
    #             self.tracker.detected_bodies(message.poses_cosy)
    #         elif message.has_poses_tracker():
    #             # message.poses_tracker contains poses from the prev time
    #             # if images in order, the bodies need no update
    #             # update only bodies that are avialable
    #             self.tracker.detected_bodies(message.poses_tracker)
    #         if message.has_image():
    #             self.tracker.set_image(message.img)
    #             self.tracker.track()
    #             # message.poses_result = self.tracker.bodies
    #             message.poses_result = {}
    #             for name, body_trans in self.tracker.get_current_preds().items():
    #                 assert body_trans.shape == (4,4)
    #                 # assert isinstance(body, Body)
    #                 message.poses_result[name] = body_trans
    #             # self.tracker.update_viewers()
    #             logging.info(f"Tracking on image {message.img_id} complete.")
    #             if self.STRIP_IMAGE:
    #                 message.img = None
    #             self.send(sender, message)




if __name__ == "__main__":

    system = ActorSystem('multiprocTCPBase', logDefs=logcfg)

    img_streamer = system.createActor(ImageStreamerActor)
    dispatcher = system.createActor(DispatcherActor)
    image_buffer = system.createActor(ImageBuffer)
    localizer = system.createActor(LocalizerActor)
    tracker_1 = system.createActor(TrackerActor)
    result_logger = system.createActor(ResultLoggerActor)

    mydict = {"buffer": image_buffer,
            "localizer": localizer,
            "tracker": tracker_1,
            "result_logger": result_logger,
            "dispatcher": dispatcher}
    cfg = ActorConfig(addressbook=mydict)

    system.tell(dispatcher, cfg)
    system.tell(localizer, cfg)


    system.tell(img_streamer, ActorConfig({"dispatcher": dispatcher}))
    system.tell(img_streamer, "start")

    time.sleep(0.5)

    system.tell(localizer, "poll")
        # time.sleep(20.0)
        

    #     time.sleep(3.0)
    #     while True:
    #         # res = system.ask(result_logger, "print")
    #         # print(res.keys())
    #         time.sleep(3.0)
    # finally:
    #     system.shutdown()

    
    #     print("shutdown complete.")
