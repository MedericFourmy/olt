#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-09-6
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#


import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from olt.config import EvaluationBOPConfig
from olt.continuous_tracker import ContinuousTracker
from olt.evaluation_tools import BOPDatasetReader
from olt.rate import Rate


def gt_predictor(sid, vid):
    time.sleep(0.4)
    return reader.predict_gt(sid=sid, vid=vid)


if __name__ == "__main__":
    eval_cfg = EvaluationBOPConfig()
    eval_cfg.ds_name = "ycbv"
    eval_cfg.tracker_cfg.tikhonov_parameter_rotation = 2000.0
    eval_cfg.localizer_cfg.detector_threshold = 0.6
    eval_cfg.tracker_cfg.viewer_display = False

    reader = BOPDatasetReader(
        eval_cfg.ds_name, load_depth=eval_cfg.tracker_cfg.use_depth
    )

    sid = list(sorted(reader.map_sids_vids.keys()))[0]
    vid = reader.map_sids_vids[sid][0]

    continuous_tracker = ContinuousTracker(
        eval_cfg=eval_cfg,
        intrinsics=reader.get_intrinsics(sid, vid),
        K=reader.get_Kres(sid, vid)[0],
        # gt_predictor=gt_predictor, # uncoment to use GT instead of cosypose
    )

    # initialize
    continuous_tracker(reader.get_obs(sid, vid).rgb, sid, vid)

    n = 100
    rate = Rate(frequency=30)
    vids = reader.map_sids_vids[sid][:n]
    # preload images as this is quite slow IO operation
    imgs = [reader.get_obs(sid, vid).rgb for vid in vids]

    processing_times = []
    for vid, img in tqdm(zip(vids, imgs)):
        start_time = time.time()
        out = continuous_tracker(img, sid, vid)
        end_time = time.time()
        processing_times.append(1000 * (end_time - start_time))
        rate.sleep()

    continuous_tracker.finish()

    fig, ax = plt.subplots(1, 1, squeeze=True)  # type: plt.Figure, plt.Axes
    ax.plot(vids, processing_times, "-o", color="tab:blue")
    ax.set_xlabel("Image id")
    ax.set_ylabel("Processing time [ms]")
    plt.show()
