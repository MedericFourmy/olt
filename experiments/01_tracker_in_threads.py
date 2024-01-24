#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-09-6
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#


import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from olt.continuous_tracker import ContinuousTracker
from olt.config import EvaluationBOPConfig
from olt.evaluation_tools import BOPDatasetReader
from olt.rate import Rate



if __name__ == "__main__":
    USE_GT_FOR_LOCALIZATION = False
    USE_DEPTH = True
    FREQ = 30
    FAKE_LOCALIZATION_DELAY = 0.4 if USE_GT_FOR_LOCALIZATION else 0.0

    eval_cfg = EvaluationBOPConfig()
    eval_cfg.ds_name = "ycbv"
    eval_cfg.tracker_cfg.tikhonov_parameter_rotation = 2000.0
    eval_cfg.localizer_cfg.detector_threshold = 0.6
    eval_cfg.tracker_cfg.viewer_display = False
    eval_cfg.tracker_cfg.use_depth = USE_DEPTH

    reader = BOPDatasetReader(
        eval_cfg.ds_name, load_depth=eval_cfg.tracker_cfg.use_depth
    )

    sid = list(sorted(reader.map_sids_vids.keys()))[0]
    vid = reader.map_sids_vids[sid][0]

    color_intrinsics = reader.get_intrinsics(sid, vid)
    depth_intrinsics = color_intrinsics if USE_DEPTH else None  # same for YCBV

    continuous_tracker = ContinuousTracker(
        tracker_cfg=eval_cfg.tracker_cfg,
        localizer_cfg=eval_cfg.localizer_cfg,
        ds_name=eval_cfg.ds_name,
        color_intrinsics=color_intrinsics,
        depth_intrinsics=depth_intrinsics,
        collect_statistics=True,
        fake_localization_delay=FAKE_LOCALIZATION_DELAY
    )

    # initialize
    object_poses = reader.predict_gt(sid, vid) if USE_GT_FOR_LOCALIZATION else None
    obs = reader.get_obs(sid, vid)
    depth = obs.depth if USE_DEPTH else None
    continuous_tracker(obs.rgb, depth, object_poses, sid, vid)

    n = 1000
    # n = -1
    rate = Rate(frequency=FREQ)
    vids = reader.map_sids_vids[sid][:n]
    # preload images as this is quite slow IO operation
    observations = [reader.get_obs(sid, vid) for vid in vids]

    processing_times = []
    for vid, obs in tqdm(zip(vids, observations)):
        start_time = time.time()
        
        object_poses = reader.predict_gt(sid, vid) if USE_GT_FOR_LOCALIZATION else None
        depth = obs.depth if USE_DEPTH else None
        continuous_tracker(obs.rgb, depth, object_poses, sid, vid)
        
        end_time = time.time()
        processing_times.append(1000 * (end_time - start_time))
        rate.sleep()

    continuous_tracker.finish()

    fig, ax = plt.subplots(1, 1, squeeze=True)
    ax.plot(vids, processing_times, "-o", color="tab:blue")
    ax.set_xlabel("Image id")
    ax.set_ylabel("Processing time [ms]")

    x = [v[-1] for v in continuous_tracker.stats_updates_from_localizer]
    miny = min(processing_times)
    maxy = max(processing_times)
    ax.vlines(x, ymin=[miny] * len(x), ymax=[maxy] * len(x), color="tab:green")
    plt.show()
