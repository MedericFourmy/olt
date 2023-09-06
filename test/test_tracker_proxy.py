
import logging

import time

import numpy as np
from olt.actor_tracker import ResultLoggerActor, TrackerRequest
from olt.tracker_proxy import TrackerProxy
import pytest
from hypothesis import given, settings, strategies as st

from thespian.actors import ActorSystem

import matplotlib.pyplot as plt

import pickle


logger = logging.getLogger("test_perf.log")

def verify_output(poses, gt=None):
    assert isinstance(poses, dict)
    assert len(poses.keys()) > 0
    assert all([pose.shape == (4,4) for pose in poses.values()])

@pytest.fixture
def tracker_proxy():
    tp = TrackerProxy()
    poses = tp.warmup_localizer()
    # tp.feed_image(TrackerRequest._get_sample_img_msg(42))
    tp._stream_imgs(True)
    time.sleep(1.0)
    tp._trigger_localizer_polling()
    assert isinstance(poses, dict)
    assert len(poses.keys()) > 0
    assert all([pose.shape == (4,4) for pose in poses.values()])
    yield tp
    try: 
        tp.shutdown()
    except AttributeError as e:
        pass


# def test_startup_shutdown(tracker_proxy):
#     tp = tracker_proxy
#     assert isinstance(tp, TrackerProxy)
#     poses = tp.warmup_localizer()

#     assert isinstance(poses, dict)
#     assert len(poses.keys()) > 0
#     assert all([pose.shape == (4,4) for pose in poses.values()])

#     tp.shutdown()

    
# def test_get_estimate(tracker_proxy):
#     tp = tracker_proxy
#     assert isinstance(tp, TrackerProxy)

    
#     req = TrackerRequest._get_sample_img_msg(250)
#     # tp.feed_image(req)


#     poses = tp.get_estimate(req, 10.0)
    
#     verify_output(poses.poses_tracker)


def test_stream_preds(tracker_proxy):
    tp = tracker_proxy
    assert isinstance(tp, TrackerProxy)

    tp._stream_imgs(True)

    time.sleep(5.0)

    preds = []
    running = True
    current_img_id = -1
    while running:
        assert isinstance(current_img_id, int)
        t = time.time()
        pred = tp.get_estimate(timeout=30.0, min_id=current_img_id+1)
        t1 = time.time()
        logger.info(f'Pred took {t1 - t}')
        if pred is None:
            pytest.fail("the pred should not be None")
        assert isinstance(pred, TrackerRequest)
        logger.info(f"Time from image capture to result: {pred.result_log_time - pred.img_time}")
        logger.info(f"Time from image capture to output: {t1 - pred.img_time}")

        current_img_id = pred.img_id
        preds.append(pred)
        if len(preds) > 20:
            break
        # if time.time() - t0 > 10.0:
        #     break

    tp._stream_imgs(False)

    
    print(preds)

    print(len(preds))
    assert len(preds) >= 5
    assert all([isinstance(pred, TrackerRequest) for pred in preds])
    assert all([pred.img_time >= 0.0 for pred in preds])
    assert preds[0].result_log_time >= preds[0].img_time
    
    for pred in preds:
        assert isinstance(pred, TrackerRequest)
        print(pred.result_log_time - pred.img_time)

def test_res_logger_none():
    system = ActorSystem()

    res_logger = system.createActor(ResultLoggerActor)

    prev_res = TrackerRequest(img_id=40, poses_result={"body42": np.eye(4)})
    system.tell(res_logger, prev_res)
    result = system.ask(res_logger, "latest_estimate")
    assert result == prev_res

    msg = TrackerRequest()
    msg.img_id = 42
    msg.timeout_policy = "none"
    msg.timeout = 0.5

    t0 = time.time()
    res = system.ask(res_logger, msg, timeout=msg.timeout+0.05)
    t1 = time.time()
    
    assert msg.timeout <= t1 - t0 <= msg.timeout+0.05
    assert res is None


    msg = TrackerRequest()
    msg.img_id = 45
    msg.timeout_policy = "none"
    msg.timeout = 0.5

    req_msg = TrackerRequest()
    req_msg.img_id = msg.img_id
    req_msg.poses_result = {"body45": np.eye(4)}


    t0 = time.time()
    system.tell(res_logger, msg)
    time.sleep(msg.timeout*0.5)
    system.tell(res_logger, req_msg)
    res = system.listen(timeout=msg.timeout*2)
    t1 = time.time()
    
    assert msg.timeout * 0.5 <= t1 - t0 <= msg.timeout+0.05
    assert res == req_msg

    system.shutdown()
@settings(deadline=5000.0)
@given(st.integers(45, 46))
def test_res_logger_newest(img_id):
    ActorSystem().shutdown()
    system = ActorSystem()

    res_logger = system.createActor(ResultLoggerActor)

    prev_res = TrackerRequest(img_id=40, poses_result={"body42": np.eye(4)})
    system.tell(res_logger, prev_res)
    result = system.ask(res_logger, "latest_estimate")
    assert result == prev_res

    msg = TrackerRequest()
    msg.img_id = 42
    msg.timeout_policy = "newest"
    msg.timeout = 0.5

    t0 = time.time()
    res = system.ask(res_logger, msg, timeout=msg.timeout+0.05)
    t1 = time.time()
    
    assert msg.timeout <= t1 - t0 <= msg.timeout+0.05
    assert res == prev_res


    msg = TrackerRequest()
    msg.img_id = 45
    msg.timeout_policy = "newest"
    msg.timeout = 0.5

    req_msg = TrackerRequest()
    req_msg.img_id = img_id
    req_msg.poses_result = {"body45": np.eye(4)}


    t0 = time.time()
    system.tell(res_logger, msg)
    time.sleep(msg.timeout*0.5)
    system.tell(res_logger, req_msg)
    res = system.listen(timeout=msg.timeout*2)
    t1 = time.time()
    
    assert msg.timeout * 0.5 <= t1 - t0 <= msg.timeout+0.05
    assert res == req_msg

    system.shutdown()

def trigger(msg: TrackerRequest):
    i = 0
    while i < 2:
        yield [msg.img_id > 42, (1, TrackerRequest())][i]
        i += 1

@given(st.integers(40, 50))
def test_trigger(img_id):
    req = TrackerRequest(img_id=img_id)
    it = trigger(req)
    val1 = it.__next__()
    assert isinstance(val1, bool)
    if val1:
        sender, msg = it.__next__()

        assert sender == 1
        assert isinstance(msg, TrackerRequest)


def test_get_estimate_performance(benchmark, tracker_proxy):
    tp = tracker_proxy
    assert isinstance(tp, TrackerProxy)

    tp._stream_imgs(True)

    time.sleep(5.0)

    preds = []

    def setup():
        logger.info("setup")
        time.sleep(0.5)
        
    pred = benchmark.pedantic(tp.get_estimate, setup=setup, rounds=20)

    # preds.append(pred)
    assert isinstance(pred, TrackerRequest)
    assert time.time() >= pred.result_log_time >= pred.img_time
    assert time.time() - pred.result_log_time <= 1.0
    assert pred.result_log_time - pred.img_time <= 2.0
    logger.info(f"Time from image capture to result: {pred.result_log_time - pred.img_time}")
    logger.info(f"Time from image capture to output: {time.time() - pred.img_time}")

    tp._stream_imgs(False)



def plot_delay(t, x):

    fig, ax = plt.subplots(1)
    assert isinstance(ax, plt.Axes)
    assert isinstance(fig, plt.Figure)
    ax.plot(t, x)

    ax.set_xlabel('time [s]')
    ax.set_ylabel('prediction delay [s]')

    # plt.show()
    fig.savefig(fname="delay.pdf")

    
def test_plot_delay():
    t = np.linspace(0,100,1000)
    x = np.sin(t)

    plot_delay(t, x)


def test_tracker_request_equals():
    t1 = TrackerRequest()
    t2 = TrackerRequest()

    assert t1 == t2
    
    t2.img_id = 42

    assert t1 != t2

def test_tuple_eq():
    assert (1,2) == (1,2)
    assert (1,2) != (1,3)

        
    

def test_make_pred_delay_graph(tracker_proxy):
    tp = tracker_proxy
    assert isinstance(tp, TrackerProxy)

    tp.system.tell(tp.img_streamer, "hz 10")


    pred = None
    while pred is None:
        pred = tp.get_latest_available_estimate()
        time.sleep(0.5)

    logger.info("First predictions are available. Start recording data.")
    current_img_id = pred.img_id


    preds = []
    running = True
    # current_img_id = -1
    while running:
        assert isinstance(current_img_id, int)
        t = time.time()
        # pred = tp.get_estimate_RT(img_id=current_img_id+1, timeout=1.0, timeout_policy="newest", message_delivery_time=0.1)
        pred = tp.get_latest_available_estimate()
        # if pred is None:
        #     with open("preds.pkl") as f:
        #         pickle.dump(preds, f)

        assert isinstance(pred, TrackerRequest)
        t1 = time.time()
        logger.info(f'Pred took {t1 - t}')

        if preds.__len__() == 0 or pred.img_id != preds[-1].img_id:
            preds.append(pred)
        else:
            continue

        latest_img_id = tp.get_latest_img_id()

        current_img_id = min(current_img_id+1, latest_img_id)
        if current_img_id >= 500:
            break

    t = np.array([pred.img_time for pred in preds])
    y = np.array([pred.result_log_time for pred in preds])

    y = y - t    # x should be the delay between adding the image to the system and the final prediction
    t = t - t[0] # start at t=0

    x = np.array([pred.img_id for pred in preds])

    logger.info(f"unique vals in x {len(np.unique(x))}")   


    with open("preds.pkl", "wb") as f:
        pickle.dump(preds, f)

    # plot_delay(x, y)


    
def test_plot_delay_from_pkl():
    with open("preds.pkl", "rb") as f:
        preds = pickle.load(f)

    t = np.array([pred.img_time for pred in preds])
    y = np.array([pred.result_log_time for pred in preds])

    y = y - t    # x should be the delay between adding the image to the system and the final prediction
    t = t - t[0] # start at t=0

    x = np.array([pred.img_id for pred in preds])
    plot_delay(x, y)

    print("wait")
    




def test_get_latest_result(benchmark, tracker_proxy):
    tp = tracker_proxy
    assert isinstance(tp, TrackerProxy)

    tp._stream_imgs(True)

    time.sleep(5.0)

    def setup():
        logger.info("setup")
        time.sleep(0.5)
        
    pred = benchmark.pedantic(tp.get_latest_available_estimate, setup=setup, rounds=20)

    assert isinstance(pred, TrackerRequest)
    verify_output(pred.poses_result)

    tp._stream_imgs(False)




def test_logging():
    for i in [1,2,3]:
        logger.info("hello logger!")
        time.sleep(0.5)


    

