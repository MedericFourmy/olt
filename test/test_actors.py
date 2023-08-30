

import logging
import time

import numpy as np
from olt.actor_tracker import ImageBuffer, ImageStreamerActor, ActorConfig, LocalizerActor, DispatcherActor, TrackerActor, TrackerRequest, ActorSystem, ResultLoggerActor

logcfg = { 'version': 1,
           'formatters': {
               'normal': {
                   'format': '%(levelname)-8s %(message)s'}},
           'handlers': {
               'h': {'class': 'logging.FileHandler',
                     'filename': 'test.log',
                     'formatter': 'normal',
                     'level': logging.INFO}},
           'loggers' : {
               '': {'handlers': ['h'], 'level': logging.DEBUG}}
         }


from thespian.actors import Actor
class CounterActor(Actor):
    def __init__(self, *args, **kwargs):
        self.count = 0
        self.msg_types = set([TrackerRequest])
        
        super().__init__(*args, **kwargs)

    def receiveMessage(self, message, sender): 
        if isinstance(message, tuple(self.msg_types)):
            self.count += 1
        if isinstance(message, str) and message == "count":
            self.send(sender, self.count)

def test_kill_system():
    system = ActorSystem('multiprocQueueBase', logDefs=logcfg)
    system.shutdown()


def test_img_streamer():
    try:
        system = ActorSystem('multiprocQueueBase', logDefs=logcfg)
    
        img_streamer = system.createActor(ImageStreamerActor)
        counter = system.createActor(CounterActor)
        
        index_0 = system.ask(img_streamer, "index", 0.1)
        print(index_0)

        system.tell(img_streamer, ActorConfig({"counter": counter}))

        system.tell(img_streamer, "start")

        time.sleep(4.0)
        system.tell(img_streamer, "stop")
        index_1 = system.ask(img_streamer, "index", 0.1)
        count = system.ask(counter, "count")
        assert (index_1-index_0) == count

    finally:
        system.shutdown()


    

def test_msg_dispatcher_img_streamer():


    try:
        system = ActorSystem('multiprocQueueBase', logDefs=logcfg)
        # system = ActorSystem()


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
        

        time.sleep(3.0)
        while True:
            res = system.ask(result_logger, "print")
            print(res.keys())
            time.sleep(0.5)

    finally:
        system.shutdown()


def test_msg_dispatcher():
    system = ActorSystem('multiprocQueueBase')
    # system = ActorSystem()

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



    imgs = [TrackerRequest._get_sample_img_msg(i) for i in range(1,100)]

    # init cosypose with first call
    # res = system.ask(localizer, imgs[0])
    system.tell(dispatcher, imgs[0])
    system.tell(localizer, "poll")
    time.sleep(20.0)

    for img in imgs[1:10]:
        assert isinstance(img, TrackerRequest)
        system.tell(dispatcher, img)
        time.sleep(1.0/30)
    system.tell(localizer, "poll")

    
    for img in imgs[10:]:
        assert isinstance(img, TrackerRequest)
        system.tell(dispatcher, img)
        time.sleep(1.0/30)

    time.sleep(3.0)
    res = system.ask(result_logger, "print")
    print(res.keys())


    system.shutdown()



def test_image_seq():
    
    imgs = [TrackerRequest._get_sample_img_msg(i) for i in range(1,100)]

    assert len(imgs) == 99

    system = ActorSystem()
    image_buffer = system.createActor(ImageBuffer)
    localizer = system.createActor(LocalizerActor)
    tracker_1 = system.createActor(TrackerActor)

    dispatcher = system.createActor(DispatcherActor,)

    system.shutdown()

def test_image_seq():
    
    imgs = [TrackerRequest._get_sample_img_msg(i) for i in range(1,100)]

    assert len(imgs) == 99

    system = ActorSystem()
    image_buffer = system.createActor(ImageBuffer)
    localizer = system.createActor(LocalizerActor)
    tracker_1 = system.createActor(TrackerActor)


    res = system.ask(localizer, imgs[0], 0.1)
    last_tracker_poses = {}
    for img in imgs:
        assert isinstance(img, TrackerRequest)
        res = system.ask(image_buffer, img, 0.1)
        if img.has_poses_cosy():
            res = system.ask(tracker_1, res, 0.1)
            last_tracker_poses = res.poses_result
        else:
            img.poses_tracker = last_tracker_poses
            res = system.ask(tracker_1, res, 0.1)

            # img.poses_tracker = system.ask(image_buffer, TrackerRequest(img_id=img.img_id-1)).poses_result
        # assert img is res
        time.sleep(1.0/100)

    for img in imgs:
        assert img.has_poses_result()
        assert img.has_poses_cosy() or img.has_poses_tracker
    

    system.shutdown()





def test_localizer():
    imgs = [TrackerRequest._get_sample_img_msg(i) for i in range(1,10)]

    system = ActorSystem()
    localizer = system.createActor(LocalizerActor)
    

    ti = [time.time()]
    for img in imgs:
        res = system.ask(localizer, img)
        ti.append(time.time())
        print(res)

    print(np.diff(ti))
        

    print(img.poses_cosy)

    system.shutdown()





def test_image_buffer_actor():
    system = ActorSystem()
        
    image_buffer = system.createActor(ImageBuffer)
    
    msg = TrackerRequest._get_sample_img_msg(1)
    assert isinstance(msg, TrackerRequest)
    print(msg)

    # adding image to buffer
    msg = system.ask(image_buffer, msg, 1)
    assert msg.img_id == 1 # the buffer was empty, so the first id should be 1

    # lets get the image back from the buffer
    msg.img = None
    res = system.ask(image_buffer, msg, 1)
    assert res.img is not None

    msgs = [TrackerRequest._get_sample_img_msg(i) for i in range(100)]
    assert isinstance(msgs[10], TrackerRequest)

    t1 = time.time()
    for msg in msgs:
        msg = system.ask(image_buffer, msg, 1)

    t2 = time.time()
    print(t2-t1)

    
    res = system.ask(image_buffer, "stats", 1)
    print(res)
    assert "60" in res


    # ask for image that is not in te buffer anymore
    msg = TrackerRequest(img_id=10)
    res = system.ask(image_buffer, msg, 1)
    assert res.img is None

    # ask for image that should be there
    msg = TrackerRequest(img_id=100)
    res = system.ask(image_buffer, msg, 1)
    assert isinstance(res.img, np.ndarray) 

    system.shutdown()