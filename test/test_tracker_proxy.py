


import time
from olt.actor_tracker import TrackerRequest
from olt.tracker_proxy import TrackerProxy
import pytest

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

    
def test_get_estimate(tracker_proxy):
    tp = tracker_proxy
    assert isinstance(tp, TrackerProxy)

    
    req = TrackerRequest._get_sample_img_msg(250)
    # tp.feed_image(req)


    poses = tp.get_estimate(req, 10.0)
    
    verify_output(poses.poses_tracker)
    


def test_passing():
    assert 42 == 42
def test_failure():
    assert 42 == -13