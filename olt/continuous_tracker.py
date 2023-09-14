#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-09-6
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
from __future__ import annotations

from typing import Union

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
        rgb_intrinsics: dict,
        depth_intrinsics: Union[dict,None] = None,
        color2depth_pose: Union[np.ndarray,None] = None,
        collect_statistics: bool = False,
        fake_localization_delay: float = 0.0, 
        accepted_objects: str = 'all',
        reset_after_n=0
    ) -> None:
        """
        gt_predictor: needed for GT localization, lambda f(sid, vid) -> object_poses, if
        None use localizer.
        """
        self.tracker_cfg = tracker_cfg
        self.reset_after_n = reset_after_n
        self.queue_img = Manager().Queue()
        self.queue_poses_initialization = SimpleQueue()
        self.collect_statistics = collect_statistics

        self.stats_updates_from_localizer: list[(int, int, int, int)] = []
        self._stats_last_update_from_localizer: float | None = None
        self._stats_localizer_freq: float = 0

        tracker_args = (
            OBJ_MODEL_DIRS[ds_name],
            accepted_objects,
            tracker_cfg,
            rgb_intrinsics,
            depth_intrinsics,
            color2depth_pose
        )
        self.main_tracker = Tracker(*tracker_args)
        self.main_tracker.init()

        self._initialized = False

        # start multiprocess tracking
        local_tracker_args = deepcopy(tracker_args)
        local_tracker_args[2].viewer_display = False
        local_tracker_args[2].viewer_save = False
        # local_tracker_args[2].tmp_dir_name += "2"

        K, _, _ = intrinsics2Kres(**rgb_intrinsics)

        localizer_args = (ds_name, localizer_cfg)
        localizer_predict_kwargs = dict(K=K, n_coarse=localizer_cfg.n_coarse, n_refiner=localizer_cfg.n_refiner)
        self.p = Process(
            target=self._localize_and_track,
            args=(
                self.queue_img,
                self.queue_poses_initialization,
                local_tracker_args,
                localizer_args,
                localizer_predict_kwargs,
                fake_localization_delay
            ),
        )
        self.p.start()

    def finish(self):
        self.queue_img.put((None, None, None, None, None))
        self.p.join()

    def _update_main_tracker_with_correction_process(self):
        # poses propagated from last detections + scores from last localization
        object_poses, scores, sid0, vid0, sidN, vidN = self.queue_poses_initialization.get()
        # Keep active tracks even if they go missing
        self.main_tracker.detected_bodies(object_poses, scores, reset_after_n=self.reset_after_n)
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
    ) -> dict[str, np.ndarray]:
        """Evaluate on the given image, if ground_truth_localizer is used, we need sid
        and vid as well."""


        if not self._initialized:
            print('Warming up CosyPose')
            # running the first iteration that will be slow, as we need to wait for the
            # localization results
            self.queue_img.put((img, depth, object_poses, sid, vid))
            self._update_main_tracker_with_correction_process()
            self.main_tracker.set_image(img, depth)
            self.main_tracker.track()
            self._initialized = True
            return self.main_tracker.get_current_preds()

        if not self.queue_poses_initialization.empty():
            self._update_main_tracker_with_correction_process()
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
        
        localizer = Localizer(*localizer_args) if localizer_args is not None else None

        tracker = Tracker(*tracker_args)
        tracker.init()
        while True:
            # Localize
            img, depth, object_poses, sid, vid = queue_img.get()
            if img is None:
                return
            
            scores_loca = None
            if object_poses is None:
                object_poses, scores_loca = localizer.predict(img, **localizer_predict_kwargs)
            if fake_localization_delay > 0.0:
                time.sleep(fake_localization_delay)

            sid0, vid0 = sid, vid

            # reset from scratch with all new detections
            tracker.detected_bodies(object_poses, scores_loca, reset_after_n=0)

            # Track
            while not queue_img.empty():
                img, depth, object_poses_not_used, sid, vid = queue_img.get()
                # Signal to exit subprocess
                if img is None:
                    return
                tracker.set_image(img, depth)
                tracker.track()

            # return poses from tracker and scores from localizer detections
            object_poses, _ = tracker.get_current_preds() 

            queue_poses_initialization.put(
                (object_poses, scores_loca, sid0, vid0, sid, vid)
            )



#############################################
#############################################
#############################################
#############################################
#############################################



class ContinuousTrackerCosytrack:
    def __init__(
        self,
        localizer_cfg: LocalizerConfig,
        ds_name: str,
        rgb_intrinsics: dict,
        collect_statistics: bool = False,
    ) -> None:
        """
        gt_predictor: needed for GT localization, lambda f(sid, vid) -> object_poses, if
        None use localizer.
        """
        self.queue_img = Manager().Queue()
        self.queue_poses_initialization = SimpleQueue()
        self.collect_statistics = collect_statistics

        self.stats_updates_from_localizer: list[(int, int, int, int)] = []
        self._stats_last_update_from_localizer: float | None = None
        self._stats_localizer_freq: float = 0

        self._initialized = False

        self.K, _, _ = intrinsics2Kres(**rgb_intrinsics)

        localizer_args = (ds_name, localizer_cfg)
        self.main_localizer = Localizer(*localizer_args)
        self.localizer_predict_kwargs = dict(K=self.K, n_coarse=1, n_refiner=6)
        self.p = Process(
            target=self._localize_and_track,
            args=(
                self.queue_img,
                self.queue_poses_initialization,
                localizer_args,
                self.localizer_predict_kwargs,
            ),
        )
        self.p.start()

    def finish(self):
        self.queue_img.put((None, None, None, None, None))
        self.p.join()

    def _update_main_tracker_with_correction_process(self):
        # poses propagated from last detections + scores from last localization
        self.data_TCO, extra_data, sid0, vid0, sidN, vidN = self.queue_poses_initialization.get()
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
        vid: int | None = None
    ) -> dict[str, np.ndarray]:
        """Evaluate on the given image, if ground_truth_localizer is used, we need sid
        and vid as well."""

        # self.data_TCO = None

        if not self._initialized:
            print('Warming up CosyPose')
            # running the first iteration that will be slow, as we need to wait for the
            # localization results
            self.main_localizer.predict(img, **self.localizer_predict_kwargs)
            self.queue_img.put((img, depth, object_poses, sid, vid))
            self._update_main_tracker_with_correction_process()
            self._initialized = True

        self.queue_img.put((img, depth, object_poses, sid, vid))

        data_TCO_track, extra_data = self.main_localizer.get_cosy_predictions(img, self.K, n_coarse=0, n_refiner=1, TCO_init=self.data_TCO, run_detector=False)
        return self.main_localizer.cosypreds2posesscores(data_TCO_track, extra_data)
        # return self.main_localizer.track(img, self.K, self.data_TCO, n_refiner=1)  # not working (!)

    @staticmethod
    def _localize_and_track(
        queue_img,
        queue_poses_initialization,
        localizer_args,
        localizer_predict_kwargs=None,
        fake_localization_delay=0.0
    ):
        localizer = Localizer(*localizer_args)

        while True:
            # Localize
            img, depth, object_poses, sid, vid = queue_img.get()
            if img is None:
                return

            scores_loca = None
            if object_poses is None:
                # object_poses, scores_loca = localizer.predict(img, **localizer_predict_kwargs)
                data_TCO, extra_data = localizer.get_cosy_predictions(img, **localizer_predict_kwargs)
                object_poses, scores_loca = localizer.cosypreds2posesscores(data_TCO, extra_data)
            if fake_localization_delay > 0.0:
                time.sleep(fake_localization_delay)

            sid0, vid0 = sid, vid

            # Track
            while not queue_img.empty():
                img, depth, object_poses_not_used, sid, vid = queue_img.get()
                # Signal to exit subprocess
                if img is None:
                    return
                t = time.perf_counter()
                
                # track with init from previous tracked pose
                data_TCO, extra_data = localizer.get_cosy_predictions(img, localizer_predict_kwargs['K'], n_coarse=0, n_refiner=1, TCO_init=data_TCO, run_detector=False)
                print('cosy tracking (ms)', 1000*(time.perf_counter() - t))
            
            print(f'Cosy tracker caught up vid0, vid {vid0}, {vid}')

            # remove poses that do not have satisfactory

            queue_poses_initialization.put(
                (data_TCO, extra_data, sid0, vid0, sid, vid)
            )
