


"""
See https://bop.felk.cvut.cz/challenges/bop-challenge-2019/#howtoparticipate
"""

from olt.actor_tracker import TrackerRequest
from olt.tracker_proxy import MultiTrackerProxy


if __name__ == '__main__':

    # METHOD = 'cosyonly'  # Run only cosypose on the bop19 targets
    # METHOD = 'threaded'  # Run continuous implementation of olt 
    # METHOD = 'cosyrefined' # Run cosypose + refinement of ICG 
    METHOD = 'ActorSystem'

    # METHOD = 'only_tracker_init' # When init() has to create new .bin files, a multiprocessing error may happen 


    import time
    from pathlib import Path
    import numpy as np
    if METHOD != "ActorSystem":
        from olt.localizer import Localizer
        from olt.tracker import Tracker
        from olt.continuous_tracker import ContinuousTracker
    from olt.evaluation_tools import BOPDatasetReader, append_result, run_bop_evaluation
    from olt.config import OBJ_MODEL_DIRS, EvaluationBOPConfig
    from olt.utils import create_video_from_images, obj_name2id, obj_label2name, intrinsics2Kres, get_method_name
    from olt.rate import Rate

    from bop_toolkit_lib import inout  # noqa

    # ActorSystem options:
    PRECHARGE = True


    # general options
    RUN_INFERENCE = True
    RUN_EVALUATION = True

    USE_DEPTH = True 
    SKIP_N_IMAGES = 0
    NB_IMG_RUN = -1  # all
    # NB_IMG_RUN = 50
    # Run the pose estimation (or get the groundtruth) every <LOCALIZE_EVERY> frames
    # LOCALIZE_EVERY = NB_IMG_RUN # Never again
    # LOCALIZE_EVERY = 1
    PRINT_INFO_EVERY = 60
    # USE_GT_FOR_LOCALIZATION = True
    # FAKE_LOCALIZATION_DELAY = 0.5
    USE_GT_FOR_LOCALIZATION = False
    FAKE_LOCALIZATION_DELAY = 0.0
    # IMG_FREQ = 5
    # IMG_FREQ = 10
    IMG_FREQ = 15
    # IMG_FREQ = 20
    # IMG_FREQ = 30
    # IMG_FREQ = 60
    # IMG_FREQ = 90

    eval_cfg = EvaluationBOPConfig() 
    eval_cfg.ds_name = 'ycbv'


    ########## TrackerConfig ###############
    eval_cfg.tracker_cfg.viewer_display = False
    eval_cfg.tracker_cfg.viewer_save = False

    eval_cfg.tracker_cfg.use_depth = USE_DEPTH
    eval_cfg.tracker_cfg.measure_occlusions = USE_DEPTH
    eval_cfg.tracker_cfg.tikhonov_parameter_rotation = 1000.0
    eval_cfg.tracker_cfg.tikhonov_parameter_translation = 30000.0

    # eval_cfg.tracker_cfg.n_corr_iterations = 0
    # eval_cfg.tracker_cfg.n_update_iterations = 0
    eval_cfg.tracker_cfg.n_corr_iterations = 4
    eval_cfg.tracker_cfg.n_update_iterations = 2
    # if nb of values in those lists are lower than n_corr_iterations, repeat last value
    
    # eval_cfg.tracker_cfg.region_scales: [7, 4, 2]
    # eval_cfg.tracker_cfg.region_standard_deviations: [25.0, 15.0, 10.0, 5.0]
    # eval_cfg.tracker_cfg.region_standard_deviations = [5*v for v in eval_cfg.tracker_cfg.region_standard_deviations]
    # eval_cfg.tracker_cfg.depth_standard_deviations = [0.0, 0.0, 0.0]
    # eval_cfg.tracker_cfg.depth_standard_deviations = [0.05, 0.03, 0.02]
    ###############

    ########## LocalizerConfig ###############
    eval_cfg.localizer_cfg.detector_threshold = 0.1  # low threshold -> high AR but lower precision
    eval_cfg.localizer_cfg.n_coarse = 1
    eval_cfg.localizer_cfg.n_refiner = 4
    # eval_cfg.localizer_cfg.renderer_name = 'panda'  # faster but from time to time very high runtime
    eval_cfg.localizer_cfg.renderer_name = 'bullet'  # higher AR, bit slower
    eval_cfg.localizer_cfg.training_type = 'synt+real'
    eval_cfg.localizer_cfg.n_workers = 0


    ds_name = eval_cfg.ds_name
    # ds_name = 'rotd'
    reader = BOPDatasetReader(ds_name, load_depth=USE_DEPTH)
    ###############



    all_sids = sorted(reader.map_sids_vids.keys())
    sidmax = all_sids[-1]

    all_bop19_results = []


    # NAMING EXPERIMENTS
    # !!!!! name should NOT contain '_' characters
    modality = 'rgbd' if USE_DEPTH else 'rgb'
    if METHOD == 'threaded':
        run_name = get_method_name(METHOD, 
                                   eval_cfg.localizer_cfg.training_type,
                                   eval_cfg.localizer_cfg.renderer_name,
                                   f'{IMG_FREQ}Hz',
                                   modality,
                                   f'n_points={eval_cfg.tracker_cfg.region_model.n_points}'
                                   )
        rate = Rate(IMG_FREQ)
    elif METHOD == 'cosyrefined':
        run_name = get_method_name(METHOD, 
                                   eval_cfg.localizer_cfg.training_type,
                                   eval_cfg.localizer_cfg.renderer_name,
                                   modality,
                                   f'npoints={eval_cfg.tracker_cfg.region_model.n_points}'
                                   )
        rate = Rate(IMG_FREQ)
    elif METHOD == 'cosyonly':
        run_name = get_method_name(METHOD, 
                                    eval_cfg.localizer_cfg.training_type, 
                                    eval_cfg.localizer_cfg.renderer_name, 
                                    f'nr{eval_cfg.localizer_cfg.n_refiner}')
    
    elif METHOD == 'ActorSystem':
        run_name = get_method_name(METHOD, 
                                   eval_cfg.localizer_cfg.training_type,
                                   eval_cfg.localizer_cfg.renderer_name,
                                   f'{IMG_FREQ}Hz',
                                   modality
                                    )
        rate = Rate(IMG_FREQ)

        

    elif METHOD == 'only_tracker_init':
        rgb_intrinsics = reader.get_intrinsics(all_sids[0], reader.map_sids_vids[all_sids[0]][0])
        K, height, width = intrinsics2Kres(**rgb_intrinsics)
        depth_intrinsics = rgb_intrinsics if USE_DEPTH else None  # same for YCBV
        tracker = Tracker(OBJ_MODEL_DIRS[ds_name], 'all', eval_cfg.tracker_cfg, rgb_intrinsics, depth_intrinsics, np.eye(4))
        tracker.init()
    
    else:
        raise ValueError(f'Method {METHOD} not defined')
    ##############################################################

    if RUN_INFERENCE:
        out = input(f'\nRunning {run_name} ? ([y]/n)\n')
        if out == 'n':
            exit('Exiting run script') 




        if METHOD == 'cosyrefined':
            localizer = Localizer(ds_name, eval_cfg.localizer_cfg)

            # # Warmup
            # localizer.predict(reader.get_obs(sid0, vids[0]).rgb, K, n_coarse=1, n_refiner=eval_cfg.localizer_cfg.n_refiner)

        for sid in all_sids:
            vids = reader.map_sids_vids[sid]

            rgb_intrinsics = reader.get_intrinsics(sid, vids[0])
            K, height, width = intrinsics2Kres(**rgb_intrinsics)
            depth_intrinsics = rgb_intrinsics if USE_DEPTH else None  # same for YCBV
            obs = reader.get_obs(sid, vids[0])
            depth = obs.depth if USE_DEPTH else None

            if METHOD == 'threaded':
                continuous_tracker = ContinuousTracker(
                    tracker_cfg=eval_cfg.tracker_cfg,
                    localizer_cfg=eval_cfg.localizer_cfg,
                    ds_name=eval_cfg.ds_name,
                    rgb_intrinsics=rgb_intrinsics,
                    depth_intrinsics=depth_intrinsics,
                    collect_statistics=True,
                    fake_localization_delay=FAKE_LOCALIZATION_DELAY
                )

                # Warmup and init
                object_poses = reader.predict_gt(sid, vids[0]) if USE_GT_FOR_LOCALIZATION else None
                continuous_tracker(obs.rgb, depth, object_poses, sid, vids[0])

            elif METHOD == 'cosyrefined':
                tracker = Tracker(OBJ_MODEL_DIRS[ds_name], 'all', eval_cfg.tracker_cfg, rgb_intrinsics, depth_intrinsics, np.eye(4))
                tracker.init()

                # Warmup
                localizer.predict(obs.rgb, K, n_coarse=1, n_refiner=eval_cfg.localizer_cfg.n_refiner)

            elif METHOD == 'cosyonly':
                localizer = Localizer(ds_name, eval_cfg.localizer_cfg)
                # Warmup
                localizer.predict(obs.rgb, K, n_coarse=1, n_refiner=eval_cfg.localizer_cfg.n_refiner)

            elif METHOD == 'ActorSystem':
                tp = MultiTrackerProxy()
                poses = tp.warmup_localizer()

                if PRECHARGE:
                    req = TrackerRequest(obs.rgb, img_id=0)
                    tp.feed_image(req=req, block=True)
                    tp._trigger_localizer_polling()

                tp.register_for_results()

            else:
                raise NotImplementedError(METHOD+' not implemented')


            # Preload observation for the scene as there is a bit I/O overhead
            observations = [reader.get_obs(sid, vid) for vid in vids]

            N_views = len(vids)
            dt_method = 0.0
            for i in range(N_views):
                vid = vids[i]

                if i < SKIP_N_IMAGES:
                    continue
                if i == NB_IMG_RUN:
                    break
                
                if METHOD == 'threaded':
                    obs = observations[i]
                    t = time.perf_counter()
                    depth = obs.depth if USE_DEPTH else None
                    object_poses = reader.predict_gt(sid, vid) if USE_GT_FOR_LOCALIZATION else None

                    preds, scores = continuous_tracker(obs.rgb, depth, object_poses, sid, vid)

                    # scene_id, obj_id, view_id, score, TCO, dt
                    dt_method = time.perf_counter() - t
                    for obj_name, TCO in preds.items():
                        score = scores[obj_name] if scores is not None else 1.0
                        # print('  obj_name, score: ', obj_name, score)
                    if reader.check_if_in_bop19_targets(sid, vid):
                        for obj_name, TCO in preds.items():
                            score = scores[obj_name] if scores is not None else 1.0
                            # print('  obj_name, score: ', obj_name, score)
                            append_result(all_bop19_results, sid, obj_name2id(obj_name), vid, score, TCO, dt_method)

                    rate.sleep()
                
                elif METHOD == 'cosyrefined':
                    if reader.check_if_in_bop19_targets(sid, vid):
                        obs = observations[i]
                        t = time.perf_counter()
                        poses, scores = localizer.predict(obs.rgb, K, n_coarse=1, n_refiner=eval_cfg.localizer_cfg.n_refiner)
                        
                        # Reset current estimations and run track
                        tracker.detected_bodies(poses, scores)
                        depth = obs.depth if USE_DEPTH else None
                        tracker.set_image(obs.rgb, depth)
                        toto = time.perf_counter()
                        tracker.track()
                        print('Tack took', 1000*(time.perf_counter() - toto))


                        poses, scores = tracker.get_current_preds()
                        dt_method = time.perf_counter() - t
                        for obj_name in poses:
                            append_result(all_bop19_results, sid, obj_name2id(obj_name), vid, scores[obj_name], poses[obj_name], dt_method)

                elif METHOD == 'cosyonly':
                    if reader.check_if_in_bop19_targets(sid, vid):
                        obs = observations[i]
                        t = time.perf_counter()
                        poses, scores = localizer.predict(obs.rgb, K, n_coarse=1, n_refiner=eval_cfg.localizer_cfg.n_refiner)

                        dt_method = time.perf_counter() - t
                        for obj_name in poses:
                            append_result(all_bop19_results, sid, obj_name2id(obj_name), vid, scores[obj_name], poses[obj_name], dt_method)

                elif METHOD == 'ActorSystem':
                    obs = observations[i]
                    t = time.perf_counter()

                    if i == 0 and not PRECHARGE:
                        tp.feed_image(TrackerRequest(img=obs.rgb, img_id=i), block=(i==0))
                        tp._trigger_localizer_polling()
                    else:
                        tp.feed_image(TrackerRequest(img=obs.rgb, img_id=i), block=False)


                    while True:
                        if i > 0:
                            res = tp.listen_for_results()
                        else: 
                            res = tp.get_latest_available_estimate(wait_for_valid_res=True)
                        assert isinstance(res, TrackerRequest)
                        if res.img_id >= i:
                            break

                    preds = res.poses_result

                    # scene_id, obj_id, view_id, score, TCO, dt
                    dt_method = time.perf_counter() - t
                    scores = None
                    for obj_name, TCO in preds.items():
                        score = scores[obj_name] if scores is not None else 1.0
                        # print('  obj_name, score: ', obj_name, score)
                    if reader.check_if_in_bop19_targets(sid, vid):
                        for obj_name, TCO in preds.items():
                            score = scores[obj_name] if scores is not None else 1.0
                            # print('  obj_name, score: ', obj_name, score)
                            append_result(all_bop19_results, sid, obj_name2id(obj_name), vid, score, TCO, dt_method)

                    rate.sleep()


                if i % PRINT_INFO_EVERY == 0:
                    print(f'Scene: {sid}/{sidmax}, View: {vid}/{vids[-1]}')
                    print('track + update_viewers (ms)', 1000*dt_method)

            
            if METHOD == 'threaded':
                # Terminate localizer subprocess
                continuous_tracker.finish()

            if METHOD == 'ActorSystem':
                # Terminate localizer subprocess
                tp.shutdown()



    RESULTS_DIR_NAME = Path('results')
    EVALUATIONS_DIR_NAME = Path('evaluations')
    RESULTS_DIR_NAME.mkdir(exist_ok=True)

    # bop result file name stricly formatted:<method>_<ds_name>-<split>.csv
    result_bop_eval_filename = f'{run_name}_{ds_name}-test.csv'
    result_bop_eval_path = RESULTS_DIR_NAME / result_bop_eval_filename

    if RUN_INFERENCE:
        inout.save_bop_results(result_bop_eval_path.as_posix(), all_bop19_results)

    if RUN_EVALUATION:
        EVALUATIONS_DIR_NAME.mkdir(exist_ok=True)
        print(f'\nEvaluating {run_name}\n')
        run_bop_evaluation(result_bop_eval_filename, RESULTS_DIR_NAME, EVALUATIONS_DIR_NAME)

    # vid_name = f'result_{eval_cfg.ds_name}_{sid}.mp4'
    # create_video_from_images(tracker.imgs_dir, vid_name=vid_name)