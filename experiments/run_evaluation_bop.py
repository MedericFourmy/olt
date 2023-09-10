


"""
See https://bop.felk.cvut.cz/challenges/bop-challenge-2019/#howtoparticipate
"""

if __name__ == '__main__':

    import time
    from pathlib import Path

    from olt.localizer import Localizer
    from olt.continuous_tracker import ContinuousTracker
    from olt.evaluation_tools import BOPDatasetReader, append_result, run_bop_evaluation
    from olt.config import OBJ_MODEL_DIRS, EvaluationBOPConfig
    from olt.utils import create_video_from_images, obj_name2id, obj_label2name, intrinsics2Kres, get_method_name
    from olt.rate import Rate

    from bop_toolkit_lib import inout  # noqa

    RUN_INFERENCE = True
    RUN_EVALUATION = True

    USE_DEPTH = False 
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
    IMG_FREQ = 5
    # IMG_FREQ = 10
    # IMG_FREQ = 15
    # IMG_FREQ = 20
    # IMG_FREQ = 30
    # IMG_FREQ = 60


    eval_cfg = EvaluationBOPConfig() 
    eval_cfg.ds_name = 'ycbv'

    eval_cfg.tracker_cfg.viewer_display = False
    eval_cfg.tracker_cfg.viewer_save = False

    eval_cfg.tracker_cfg.use_depth = USE_DEPTH
    eval_cfg.tracker_cfg.measure_occlusions = USE_DEPTH
    eval_cfg.tracker_cfg.tikhonov_parameter_rotation = 5000.0
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

    eval_cfg.localizer_cfg.detector_threshold = 0.1
    eval_cfg.localizer_cfg.n_coarse = 1
    eval_cfg.localizer_cfg.n_refiner = 4
    # eval_cfg.localizer_cfg.renderer_name = 'panda'  # faster but from time to time very high runtime
    eval_cfg.localizer_cfg.renderer_name = 'bullet'  # higher AR, bit slower
    eval_cfg.localizer_cfg.training_type = 'synt+real'
    eval_cfg.localizer_cfg.n_workers = 0


    ds_name = eval_cfg.ds_name
    # ds_name = 'rotd'
    reader = BOPDatasetReader(ds_name, load_depth=USE_DEPTH)



    all_sids = sorted(reader.map_sids_vids.keys())
    sidmax = all_sids[-1]

    all_bop19_results = []

    # METHOD = 'cosyonly'
    METHOD = 'threaded'

    # NAMING EXPERIMENTS
    modality = 'rgbd' if USE_DEPTH else 'rgb'
    # COSYPOSE_ONLY = True
    if METHOD == 'threaded':
        run_name = get_method_name(METHOD, 
                                      eval_cfg.localizer_cfg.training_type,
                                      eval_cfg.localizer_cfg.renderer_name,
                                      f'{IMG_FREQ}Hz',
                                      modality)
        rate = Rate(IMG_FREQ)

    elif METHOD == 'cosyonly':
        run_name = get_method_name(METHOD, 
                                    eval_cfg.localizer_cfg.training_type, 
                                    eval_cfg.localizer_cfg.renderer_name, 
                                    f'nr{eval_cfg.localizer_cfg.n_refiner}')
    ##############################################################

    if RUN_INFERENCE:
        out = input(f'\nRunning {run_name} ? ([y]/n)\n')
        if out == 'n':
            exit('Exiting run script') 

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

            elif METHOD == 'cosyonly':
                localizer = Localizer(ds_name, eval_cfg.localizer_cfg)
                # Warmup
                localizer.predict(obs.rgb, K, n_coarse=1, n_refiner=eval_cfg.localizer_cfg.n_refiner)


            # Preload observation for the scene as the is a bit I/O costly
            # observations = [reader.get_obs(sid, vid) for vid in vids]

            N_views = len(vids)
            dt_method = 0.0
            for i in range(N_views):
                vid = vids[i]

                if i < SKIP_N_IMAGES:
                    continue
                if i == NB_IMG_RUN:
                    break
                
                if METHOD == 'threaded':
                    obs = reader.get_obs(sid, vid)
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

                elif METHOD == 'cosyonly':
                    if reader.check_if_in_bop19_targets(sid, vid):
                        obs = reader.get_obs(sid, vid)
                        t = time.perf_counter()
                        # preds, scores = localizer.predict(obs.rgb, K, n_coarse=1, n_refiner=eval_cfg.localizer_cfg.n_refiner)
                        data_TCO, extra_data = localizer.get_cosy_predictions(obs.rgb, K, n_coarse=1, n_refiner=eval_cfg.localizer_cfg.n_refiner)
                        poses = data_TCO.poses.cpu()

                        dt_method = time.perf_counter() - t
                        for k in range(len(data_TCO)):
                            obj_label = data_TCO.infos['label'][k]
                            obj_name = obj_label2name(obj_label, ds_name)
                            score = data_TCO.infos['score'][k]
                            TCO = poses[k].numpy()

                            append_result(all_bop19_results, sid, obj_name2id(obj_name), vid, score, TCO, dt_method)

                if i % PRINT_INFO_EVERY == 0:
                    print(f'Scene: {sid}/{sidmax}, View: {vid}/{vids[-1]}')
                    print('track + update_viewers (ms)', 1000*dt_method)

            
            if METHOD == 'threaded':
                # Terminate localizer subprocess
                continuous_tracker.finish()



    RESULTS_DIR_NAME = Path('results')
    EVALUATIONS_DIR_NAME = Path('evaluations')
    RESULTS_DIR_NAME.mkdir(exist_ok=True)

    # bop result file name stricly formatted:<method>_<ds_name>-<split>.csv
    result_bop_eval_filename = f'{run_name}_{ds_name}-test.csv'

    if RUN_INFERENCE:
        inout.save_bop_results(f'{RESULTS_DIR_NAME.as_posix()}/{result_bop_eval_filename}', all_bop19_results)

    if RUN_EVALUATION:
        EVALUATIONS_DIR_NAME.mkdir(exist_ok=True)
        print(f'\Evaluating {run_name}\n')
        run_bop_evaluation(result_bop_eval_filename, RESULTS_DIR_NAME, EVALUATIONS_DIR_NAME)

    # vid_name = f'result_{eval_cfg.ds_name}_{sid}.mp4'
    # create_video_from_images(tracker.imgs_dir, vid_name=vid_name)