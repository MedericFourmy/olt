import subprocess
import time

import pytest
from olt.actor_tracker import LocalizerActor
from olt.actor_tracker import TrackerRequest
from olt.config import logcfg

import logging

from thespian.actors import *

import numpy as np

from pynvml import *

logger = logging.getLogger("test_logger")

@pytest.fixture(autouse=True)
def check_gpu_mem():
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    logger.info(f'total    : {info.total}')
    logger.info(f'free     : {info.free}')
    logger.info(f'used     : {info.used}')

    assert info.free > 2156686336

    yield info
    subprocess.run("kill -9 $(ps ax | grep Actor | fgrep -v grep | awk '{ print $1 }')", shell=True)

    

def test_check_gpu_mem_fixture(check_gpu_mem):

    print(check_gpu_mem.free)
    assert check_gpu_mem.free > 3156686336

def shutdown(system):
    # localizer_pid = self.system.ask(self.localizer, 'exit')
    # os.kill(localizer_pid, 9)
    system.shutdown()
    time.sleep(0.5)
    subprocess.run("kill -9 $(ps ax | grep Actor | fgrep -v grep | awk '{ print $1 }')", shell=True)


@pytest.fixture
def loc_system():

    system = ActorSystem('multiprocTCPBase', logDefs=logcfg)

    localizer = system.createActor(LocalizerActor)

    ## warmup
    img = TrackerRequest._get_sample_img_msg(42)
    assert isinstance(img, TrackerRequest)
    assert img.has_image()
    poses = system.ask(localizer, img, 25.0)

    assert isinstance(poses, dict)
    assert len(poses.keys()) > 0
    assert all([pose.shape == (4,4) for pose in poses.values()])

    
    assert isinstance(poses, dict)
    assert len(poses.keys()) > 0
    assert all([pose.shape == (4,4) for pose in poses.values()])
    yield (system, localizer)
    try: 
        shutdown(system)
    except AttributeError as e:
        pass





def test_localizer_warmup(benchmark):
    system = ActorSystem('multiprocTCPBase', logDefs=logcfg)

    localizer = system.createActor(LocalizerActor)

    img = TrackerRequest._get_sample_img_msg(42)
    assert isinstance(img, TrackerRequest)
    assert img.has_image()
    poses = system.ask(localizer, img, 25.0)

    assert isinstance(poses, dict)
    assert len(poses.keys()) > 0
    assert all([pose.shape == (4,4) for pose in poses.values()])

    def predict(img):
        poses = system.ask(localizer, img, 25.0)
        assert isinstance(poses, dict)
        assert len(poses.keys()) > 0
        assert all([pose.shape == (4,4) for pose in poses.values()])
        return poses


    poses = benchmark(predict, img)

    assert isinstance(poses, dict)
    assert len(poses.keys()) > 0
    assert all([pose.shape == (4,4) for pose in poses.values()])

    system.shutdown()


def test_localizer_performance(benchmark, loc_system):
    # system = ActorSystem('multiprocTCPBase')

    # localizer = system.createActor(LocalizerActor)
    system, localizer = loc_system

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



