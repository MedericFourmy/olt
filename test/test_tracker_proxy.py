
import logging

import time
from olt.actor_tracker import TrackerRequest
from olt.tracker_proxy import TrackerProxy
import pytest

logger = logging.getLogger("test_perf.log")

def verify_output(poses, gt=None):
    assert isinstance(poses, dict)
    assert len(poses.keys()) > 0
    assert all([pose.shape == (4,4) for pose in poses.values()])

@pytest.fixture
def tracker_proxy():
    tp = TrackerProxy()
    poses = tp.warmup_localizer()
    tp.feed_image(TrackerRequest._get_sample_img_msg(42))
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


    

