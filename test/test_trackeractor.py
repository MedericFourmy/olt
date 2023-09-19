import subprocess
import time

import pytest
from olt.actor_tracker import TrackerActor
from olt.actor_tracker import TrackerRequest
from olt.evaluation_tools import BOPDatasetReader
from thespian.actors import *

import numpy as np

DS_NAME = 'ycbv'
SCENE_ID = 48
VIEW_ID = 1


def shutdown(system):
    # localizer_pid = self.system.ask(self.localizer, 'exit')
    # os.kill(localizer_pid, 9)
    system.shutdown()
    time.sleep(0.5)
    subprocess.run("kill -9 $(ps ax | grep Actor | fgrep -v grep | awk '{ print $1 }')", shell=True)


@pytest.fixture
def bop_ds():

    bop_ds = BOPDatasetReader(ds_name=DS_NAME)
    yield bop_ds


@pytest.fixture
def track_system():

    system = ActorSystem('multiprocTCPBase')

    tracker = system.createActor(TrackerActor)

    yield (system, tracker)
    try: 
        shutdown(system)
    except AttributeError as e:
        pass


def test_tracker_performance(benchmark, track_system, bop_ds):
    # system = ActorSystem('multiprocTCPBase')

    # localizer = system.createActor(LocalizerActor)
    system, localizer = track_system

    assert isinstance(bop_ds, BOPDatasetReader)
    gt = bop_ds.predict_gt(sid=SCENE_ID, vid=VIEW_ID)

    

    def predict(img):
        poses = system.ask(localizer, img, 25.0)
        assert isinstance(poses, dict)
        assert len(poses.keys()) > 0
        assert all([pose.shape == (4,4) for pose in poses.values()])
        return poses

    def setup():
        img_nr = np.random.randint(1,2200)
        img = TrackerRequest._get_sample_img_msg(img_nmb=img_nr)
        return [img], {}

    poses = benchmark.pedantic(predict, setup=setup, rounds=50)

    assert isinstance(poses, dict)
    assert len(poses.keys()) > 0
    assert all([pose.shape == (4,4) for pose in poses.values()])




