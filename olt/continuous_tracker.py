#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-09-6
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
from __future__ import annotations

import time
from copy import deepcopy
from multiprocessing import SimpleQueue, Process, Manager

import numpy as np

from olt.config import EvaluationBOPConfig, OBJ_MODEL_DIRS
from olt.localizer import Localizer
from olt.tracker import Tracker


class ContinuousTracker:
    def __init__(
        self,
        eval_cfg: EvaluationBOPConfig,
        intrinsics,
        K,
        gt_predictor=None,
        collect_statistics=False,
    ) -> None:
        """
        gt_predictor: needed for GT localization, lambda f(sid, vid) -> predictions, if
        None use localizer.
        """
        super().__init__()
        self.eval_cfg = eval_cfg
        self.queue_img = Manager().Queue()
        self.queue_poses_initialization = SimpleQueue()
        self.collect_statistics = collect_statistics

        self.stats_updates_from_localizer: list[(int, int, int, int)] = []
        self._stats_last_update_from_localizer: float | None = None
        self._stats_localizer_freq: float = 0

        tracker_args = (
            OBJ_MODEL_DIRS[eval_cfg.ds_name],
            "all",
            eval_cfg.tracker_cfg,
            intrinsics,
        )
        self.main_tracker = Tracker(*tracker_args)
        self.main_tracker.init()

        self._initialized = False

        # start multiprocess tracking
        local_tracker_args = deepcopy(tracker_args)
        local_tracker_args[2].tmp_dir_name += "2"
        if gt_predictor is None:
            localizer_args = (eval_cfg.ds_name, eval_cfg.localizer_cfg)
            localizer_predict_kwargs = dict(K=K, n_coarse=1, n_refiner=6)
        else:
            print("Warning: you are using GT predictions")
            localizer_args, localizer_predict_kwargs = None, None
        self.p = Process(
            target=self._localize_and_track,
            args=(
                self.queue_img,
                self.queue_poses_initialization,
                local_tracker_args,
                gt_predictor,
                localizer_args,
                localizer_predict_kwargs,
            ),
        )
        self.p.start()

    def finish(self):
        self.queue_img.put((None, None, None))
        self.p.join()

    def _update_main_tracker_from_localizer(self):
        predictions, sid0, vid0, sidN, vidN = self.queue_poses_initialization.get()
        self.main_tracker.detected_bodies(predictions)
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
        sid: int | None = None,
        vid: int | None = None,
        without_localizer: bool = False,
    ) -> dict[str, np.ndarray]:
        """Evaluate on the given image, if ground_truth_localizer is used, we need sid
        and vid as well."""

        if not self._initialized:
            # running the first iteration that will be slow, as we need to wait for the
            # localization results
            self.queue_img.put((img, sid, vid))
            self._update_main_tracker_from_localizer()
            self.main_tracker.set_image(img)
            self.main_tracker.track()
            self._initialized = True
            return self.main_tracker.get_current_preds()

        if not without_localizer:
            if not self.queue_poses_initialization.empty():
                self._update_main_tracker_from_localizer()
            self.queue_img.put((img, sid, vid))

        self.main_tracker.set_image(img)
        self.main_tracker.track()

        if self.eval_cfg.tracker_cfg.viewer_display:
            self.main_tracker.update_viewers()

        return self.main_tracker.get_current_preds()

    @staticmethod
    def _localize_and_track(
        queue_img,
        queue_poses_initialization,
        tracker_args,
        predict_gt=None,
        localizer_args=None,
        localizer_predict_kwargs=None,
    ):
        """Use GT prediction to fake localizer, then run tracker until the pipe queue
        is empty, send results to main thread and run localizer again."""
        assert predict_gt is not None or localizer_args is not None
        assert predict_gt is None or localizer_args is None
        localizer = Localizer(*localizer_args) if localizer_args is not None else None

        tracker = Tracker(*tracker_args)
        tracker.init()
        while True:
            # Localize
            img, sid, vid = queue_img.get()
            if img is None:
                return

            if predict_gt is not None:
                predictions = predict_gt(sid=sid, vid=vid)
            else:
                predictions = localizer.predict(img, **localizer_predict_kwargs)
            tracker.detected_bodies(predictions)

            sid0, vid0 = sid, vid

            # Track
            while not queue_img.empty():
                img, sid, vid = queue_img.get()
                if img is None:
                    return
                tracker.set_image(img)
                tracker.track()

            queue_poses_initialization.put(
                (tracker.get_current_preds(), sid0, vid0, sid, vid)
            )
