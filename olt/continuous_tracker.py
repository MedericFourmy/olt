#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-09-6
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
from __future__ import annotations

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
from multiprocessing import SimpleQueue, Process, Manager

import time
from copy import deepcopy

import numpy as np

from olt.config import TrackerConfig, LocalizerConfig, OBJ_MODEL_DIRS
from olt.localizer import Localizer
from olt.tracker import Tracker
from olt.utils import intrinsics2Kres



class ContinuousTracker:
    def __init__(
        self,
        tracker_cfg: TrackerConfig,
        localizer_cfg: LocalizerConfig,
        ds_name: str,
        rgb_intrinsics,
        depth_intrinsics=None,
        collect_statistics=False,
        fake_localization_delay=0.0
    ) -> None:
        """
        gt_predictor: needed for GT localization, lambda f(sid, vid) -> object_poses, if
        None use localizer.
        """
        self.tracker_cfg = tracker_cfg
        self.queue_img = Manager().Queue()
        self.queue_poses_initialization = SimpleQueue()
        self.collect_statistics = collect_statistics
        self.fake_localization_delay = fake_localization_delay

        self.stats_updates_from_localizer: list[(int, int, int, int)] = []
        self._stats_last_update_from_localizer: float | None = None
        self._stats_localizer_freq: float = 0

        tracker_args = (
            OBJ_MODEL_DIRS[ds_name],
            "all",
            tracker_cfg,
            rgb_intrinsics,
            depth_intrinsics
        )
        self.main_tracker = Tracker(*tracker_args)
        self.main_tracker.init()

        self._initialized = False

        # start multiprocess tracking
        local_tracker_args = deepcopy(tracker_args)
        local_tracker_args[2].tmp_dir_name += "2"

        K, _, _ = intrinsics2Kres(**rgb_intrinsics)

        localizer_args = (ds_name, localizer_cfg)
        localizer_predict_kwargs = dict(K=K, n_coarse=1, n_refiner=6)
        self.p = Process(
            target=self._localize_and_track,
            args=(
                self.queue_img,
                self.queue_poses_initialization,
                local_tracker_args,
                localizer_args,
                localizer_predict_kwargs,
                self.fake_localization_delay
            ),
        )
        self.p.start()

    def finish(self):
        self.queue_img.put((None, None, None, None, None))
        self.p.join()

    def _update_main_tracker_from_localizer(self):
        object_poses, sid0, vid0, sidN, vidN = self.queue_poses_initialization.get()
        self.main_tracker.detected_bodies(object_poses)
        if self.collect_statistics:
            if self._stats_last_update_from_localizer is not None:
                self._stats_localizer_freq = 1.0 / (
                    time.time() - self._stats_last_update_from_localizer
                )
            self._stats_last_update_from_localizer = time.time()
            self.stats_updates_from_localizer.append((sid0, vid0, sidN, vidN))

    def __call__(
        self,
        img: np.ndarray,
        depth: np.ndarray | None = None,
        object_poses: dict | None = None,
        sid: int | None = None,
        vid: int | None = None,
        without_localizer: bool = False,
    ) -> dict[str, np.ndarray]:
        """Evaluate on the given image, if ground_truth_localizer is used, we need sid
        and vid as well."""

        if not self._initialized:
            # running the first iteration that will be slow, as we need to wait for the
            # localization results
            self.queue_img.put((img, depth, object_poses, sid, vid))
            self._update_main_tracker_from_localizer()
            self.main_tracker.set_image(img, depth)
            self.main_tracker.track()
            self._initialized = True
            return self.main_tracker.get_current_preds()

        if not without_localizer:
            if not self.queue_poses_initialization.empty():
                self._update_main_tracker_from_localizer()
            self.queue_img.put((img, depth, object_poses, sid, vid))

        self.main_tracker.set_image(img, depth)
        self.main_tracker.track()

        if self.tracker_cfg.viewer_display or self.tracker_cfg.viewer_save:
            self.main_tracker.update_viewers()

        return self.main_tracker.get_current_preds()

    @staticmethod
    def _localize_and_track(
        queue_img,
        queue_poses_initialization,
        tracker_args,
        localizer_args=None,
        localizer_predict_kwargs=None,
        fake_localization_delay=0.0
    ):
        """Use GT prediction to fake localizer, then run tracker until the pipe queue
        is empty, send results to main thread and run localizer again."""
        localizer = Localizer(*localizer_args) if localizer_args is not None else None

        tracker = Tracker(*tracker_args)
        tracker.init()
        while True:
            # Localize
            img, depth, object_poses, sid, vid = queue_img.get()
            if img is None:
                return

            if object_poses is None:
                object_poses = localizer.predict(img, **localizer_predict_kwargs)
            if fake_localization_delay > 0.0:
                time.sleep(fake_localization_delay)
            tracker.detected_bodies(object_poses)
            tracker.set_image(img, depth)
            tracker.track()

            sid0, vid0 = sid, vid

            # Track
            while not queue_img.empty():
                img, depth, object_poses_not_used, sid, vid = queue_img.get()
                if img is None:
                    return
                tracker.set_image(img, depth)
                tracker.track()

            queue_poses_initialization.put(
                (tracker.get_current_preds(), sid0, vid0, sid, vid)
            )
