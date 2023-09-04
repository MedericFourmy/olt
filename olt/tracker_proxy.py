import os
import subprocess
import time
import logging

from thespian.actors import ActorExitRequest

import numpy as np
from olt.actor_tracker import ActorConfig, ActorSystem, DispatcherActor, ImageBuffer, ImageStreamerActor, LocalizerActor, ResultLoggerActor, TrackerActor, TrackerRequest
from olt.config import logcfg



class TrackerProxy(object):
    def __init__(self) -> None:
        self.system = ActorSystem('multiprocTCPBase', logDefs=logcfg)

        self.img_streamer = self.system.createActor(ImageStreamerActor)
        self.dispatcher =self.system.createActor(DispatcherActor)
        self.image_buffer =self.system.createActor(ImageBuffer)
        self.localizer = self.system.createActor(LocalizerActor)
        self.tracker_1 = self.system.createActor(TrackerActor)
        self.result_logger = self.system.createActor(ResultLoggerActor)

        mydict = {"buffer": self.image_buffer,
                "localizer": self.localizer,
                "tracker": self.tracker_1,
                "result_logger": self.result_logger,
                "dispatcher": self.dispatcher}
        cfg = ActorConfig(addressbook=mydict)

        self.system.tell(self.dispatcher, cfg)
        self.system.tell(self.localizer, cfg)
        self.system.tell(self.tracker_1, cfg)

        self.system.tell(self.img_streamer, ActorConfig({"buffer": self.image_buffer}))
        self.system.tell(self.img_streamer, "hz 30")

        # self.system.tell(self.image_buffer, ActorConfig({"tracker": self.image_buffer}))

        
    def warmup_localizer(self):
        img = TrackerRequest._get_sample_img_msg(42)
        poses = self.system.ask(self.localizer, img, 25.0)
        return poses

    def _stream_imgs(self, activate: bool=True):
        if activate:
            self.system.tell(self.img_streamer, "start")
            return
        elif not activate:
            self.system.tell(self.img_streamer, "stop")
            return
        
    def _trigger_localizer_polling(self):
        self.system.tell(self.localizer, "poll")


    def feed_image(self, req: (TrackerRequest, np.ndarray)) -> int:
        old_img_id = self.system.ask(self.image_buffer, "latest_image_id", 0.2)

        
        if isinstance(req, np.ndarray):
            req = TrackerRequest(img=req)

        assert isinstance(req, TrackerRequest)
        assert req.has_image()
        assert not req.has_id()
        self.system.tell(self.image_buffer, req)
        img_id = self.system.ask(self.image_buffer, "latest_image_id", 0.2)
        assert img_id > old_img_id
        assert isinstance(img_id, int)
        return img_id

    def get_latest_available_estimate(self):
        result = self.system.ask(self.result_logger, "latest_estimate")
        assert isinstance(result, TrackerRequest)
        return result



    def get_estimate(self, img: (np.ndarray, TrackerRequest) = None, timeout: float = 10.0, min_id: int = None):
        # self.feed_image(img)
        if img is None:
            img = TrackerRequest()
            img.img_id = self.system.ask(self.image_buffer, "latest_image_id", 0.2)
            assert isinstance(img.img_id, int) 
        elif isinstance(img, TrackerRequest):
            img.img_id = self.feed_image(img)
        
        if isinstance(min_id, TrackerRequest):
            print('Halt...')


        if min_id is not None:
            img.img_id = max(img.img_id, min_id)
        
        img.img = None
        logging.info(f"asking result for img_id {img.img_id}.")
        # img.img_id = self.system.ask(self.image_buffer, "latest_image_id", 0.2)
        result = self.system.ask(self.result_logger, img, timeout)
        # result = self.system.listen(timeout=timeout)
        if result is None:
            print('Halt...')
        return result


    def shutdown(self):
        # localizer_pid = self.system.ask(self.localizer, 'exit')
        # os.kill(localizer_pid, 9)
        self.system.shutdown()
        time.sleep(0.5)
        subprocess.run("kill -9 $(ps ax | grep Actor | fgrep -v grep | awk '{ print $1 }')", shell=True)

        

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