


"""
See https://bop.felk.cvut.cz/challenges/bop-challenge-2019/#howtoparticipate
"""


if __name__ == '__main__':


    # ActorSystem options:
    PRECHARGE = True

    def parse_script_input():
        import argparse
        parser = argparse.ArgumentParser(
            prog='run_evaluation_bop',
            description='Run different implementation of object localization and trackign on BOP formatted datasets'
        )

        parser.add_argument('--method',  dest='method', type=str, required=True, help='threaded, cosyonly, cosyrefined, ActorSystem, trackfromstart, only_tracker_init')
        parser.add_argument('--suffix',  dest='suffix', type=str, required=False, default='', help='append suffix to method name when saving result file')
        parser.add_argument('--overwrite',  dest='overwrite', action='store_true', default=False, help='orverwrite result files without asking')
        parser.add_argument('--no-inference',  dest='run_inference',  action='store_false', default=True)
        parser.add_argument('--no-evaluation', dest='run_evaluation', action='store_false', default=True)
        parser.add_argument('--img-freq',  dest='img_freq', type=int, default=30)
        parser.add_argument('--use-depth', dest='use_depth', action='store_true', default=False)
        parser.add_argument('--no-depth-modality', dest='no_depth_modality', action='store_true', default=False)
        parser.add_argument('--viewer-display', dest='viewer_display', action='store_true', default=False)
        parser.add_argument('--display-depth', dest='display_depth', action='store_true', default=False)
        parser.add_argument('--viewer-save', dest='viewer_save', action='store_true', default=False)
        parser.add_argument('--use-gt-for-localization', dest='use_gt_for_localization', action='store_true', default=False)
        parser.add_argument('--use-cosypose-as-tracker', dest='use_cosypose_as_tracker', action='store_true', default=False)
        parser.add_argument('--fake-localization-delay', dest='fake_localization_delay', type=float, default=0.0)
        parser.add_argument('--print-info-every', dest='print_info_every', type=int, default=60)
        parser.add_argument('--n-refiner', dest='n_refiner', type=int, default=4)
        parser.add_argument('--reset-after-n', dest='reset_after_n', type=int, default=0)
        

        return parser.parse_args()

    # args.method = 'threaded'  # Run continuous implementation of olt 
    # args.method = 'cosyonly'  # Run only cosypose on the bop19 targets
    # args.method = 'cosyrefined' # Run cosypose + refinement of ICG 
    # args.method = 'ActorSystem'
    # args.method = 'trackfromstart'
    # args.method = 'only_tracker_init' # When init() has to create new .bin files, a multiprocessing error may happen 


    args = parse_script_input()
    
    assert args.method in ['threaded', 'cosyonly', 'cosyrefined', 'ActorSystem', 'trackfromstart', 'only_tracker_init'], f'{args.method} is wrong'

    import time
    from pathlib import Path
    import numpy as np
    if args.method == "ActorSystem":
        from olt.actor_tracker import TrackerRequest
        from olt.tracker_proxy import MultiTrackerProxy
    if args.method in ['cosyonly', 'cosyrefined', 'trackfromstart']:
        from olt.tracker import Tracker
        from olt.localizer import Localizer
    if args.method == 'threaded':
        from olt.continuous_tracker import ContinuousTracker, ContinuousTrackerCosytrack
    from olt.evaluation_tools import BOPDatasetReader, append_result, run_bop_evaluation
    from olt.config import OBJ_MODEL_DIRS, EvaluationBOPConfig
    from olt.utils import create_video_from_images, obj_name2id, intrinsics2Kres, get_method_name
    from olt.rate import Rate

    from bop_toolkit_lib import inout  # noqa



    eval_cfg = EvaluationBOPConfig() 
    eval_cfg.ds_name = 'ycbv'

    ########## TrackerConfig ###############
    eval_cfg.tracker_cfg.use_depth = args.use_depth
    eval_cfg.tracker_cfg.viewer_display = args.viewer_display
    eval_cfg.tracker_cfg.display_depth = args.display_depth
    eval_cfg.tracker_cfg.viewer_save = args.viewer_save
    eval_cfg.tracker_cfg.measure_occlusions = args.use_depth
    eval_cfg.tracker_cfg.no_depth_modality = args.no_depth_modality    
    # eval_cfg.tracker_cfg.measure_occlusions = False



    if args.method in ['cosyrefined']:
    # if args.method in ['threaded', 'cosyrefined']:
        # YCBV refinement SETTINGS
        eval_cfg.tracker_cfg.n_corr_iterations = 7
        eval_cfg.tracker_cfg.n_update_iterations = 2
        eval_cfg.tracker_cfg.tikhonov_parameter_rotation = 100.0
        eval_cfg.tracker_cfg.tikhonov_parameter_translation = 30000.0
        eval_cfg.tracker_cfg.region_modality.scales = [7, 4, 2]
        eval_cfg.tracker_cfg.region_modality.standard_deviations = [25.0, 15.0, 10.0]
        eval_cfg.tracker_cfg.region_modality.n_unoccluded_iterations = 0
        eval_cfg.tracker_cfg.depth_modality.considered_distances = [0.300, 0.250, 0.100]
        eval_cfg.tracker_cfg.depth_modality.standard_deviations = [0.100, 0.05, 0.02]
        eval_cfg.tracker_cfg.depth_modality.stride_length = 0.010  # "for efficiency"
        eval_cfg.tracker_cfg.depth_modality.n_unoccluded_iterations = 0

    else:
        # YCBV tracking SETTINGS
        eval_cfg.tracker_cfg.n_corr_iterations = 4
        eval_cfg.tracker_cfg.n_update_iterations = 2
        eval_cfg.tracker_cfg.tikhonov_parameter_rotation =    600000.0
        eval_cfg.tracker_cfg.tikhonov_parameter_translation = 6000000.0
        # eval_cfg.tracker_cfg.tikhonov_parameter_rotation    = 3000000.0
        # eval_cfg.tracker_cfg.tikhonov_parameter_translation = 30000000.0
        eval_cfg.tracker_cfg.region_modality.scales = [7, 4, 2]
        eval_cfg.tracker_cfg.region_modality.standard_deviations: [25.0, 15.0, 10.0]
        eval_cfg.tracker_cfg.region_modality.n_unoccluded_iterations = 0
        eval_cfg.tracker_cfg.depth_modality.considered_distances = [0.07, 0.05, 0.04]
        eval_cfg.tracker_cfg.depth_modality.standard_deviations = [0.05, 0.03, 0.02]
        # eval_cfg.tracker_cfg.depth_modality.considered_distances = [0.300, 0.250, 0.100]
        # eval_cfg.tracker_cfg.depth_modality.standard_deviations = [0.100, 0.05, 0.02]
        eval_cfg.tracker_cfg.depth_modality.n_unoccluded_iterations = 0

        eval_cfg.tracker_cfg.depth_scale = 0.001  # bop reader returns epth in millimeter
        # eval_cfg.tracker_cfg.depth_scale = 1000.0
         


    # # DEACTIVATE TRACKER
    # eval_cfg.tracker_cfg.n_corr_iterations = 0
    # eval_cfg.tracker_cfg.n_update_iterations = 0
    
    ###############

    ########## LocalizerConfig ###############
    eval_cfg.localizer_cfg.detector_threshold = 0.01  # low threshold -> high AR but lower precision
    eval_cfg.localizer_cfg.n_coarse = 1
    eval_cfg.localizer_cfg.n_refiner = args.n_refiner
    # eval_cfg.localizer_cfg.renderer_name = 'panda'  # faster but from time to time very high runtime
    eval_cfg.localizer_cfg.renderer_name = 'bullet'  # higher AR, bit slower
    eval_cfg.localizer_cfg.training_type = 'synt+real'
    eval_cfg.localizer_cfg.n_workers = 0


    ds_name = eval_cfg.ds_name
    # ds_name = 'rotd'
    reader = BOPDatasetReader(ds_name, load_depth=args.use_depth)
    ###############



    all_sids = sorted(reader.map_sids_vids.keys())
    # all_sids = [all_sids[0]]
    # all_sids = [all_sids[2]]
    all_bop19_results = []


    # NAMING EXPERIMENTS
    # !!!!! name should NOT contain '_' characters
    modality = 'rgbd' if args.use_depth else 'rgb'
    if args.method == 'threaded':
        run_name = get_method_name(args.method, 
                                   eval_cfg.localizer_cfg.training_type,
                                   eval_cfg.localizer_cfg.renderer_name,
                                   f'{args.img_freq}Hz',
                                   f'nr{eval_cfg.localizer_cfg.n_refiner}',
                                   modality,
                                   )
        if args.use_cosypose_as_tracker:
            run_name += '-cosytrack'
        rate = Rate(args.img_freq)
    elif args.method == 'cosyrefined':
        run_name = get_method_name(args.method, 
                                   eval_cfg.localizer_cfg.training_type,
                                   eval_cfg.localizer_cfg.renderer_name,
                                   modality,
                                   f'npoints={eval_cfg.tracker_cfg.region_model.n_points}'
                                   )

    elif args.method == 'cosyonly':
        run_name = get_method_name(args.method, 
                                    eval_cfg.localizer_cfg.training_type, 
                                    eval_cfg.localizer_cfg.renderer_name, 
                                    f'nr{eval_cfg.localizer_cfg.n_refiner}'
                                    )

    elif args.method == 'trackfromstart':
        run_name = get_method_name(args.method,
                                   f'from-gt={args.use_gt_for_localization}',
                                    modality
                                    )
        
    elif args.method == 'ActorSystem':
        run_name = get_method_name(args.method, 
                                   eval_cfg.localizer_cfg.training_type,
                                   eval_cfg.localizer_cfg.renderer_name,
                                   f'{args.img_freq}Hz',
                                   modality
                                    )
        rate = Rate(args.img_freq)

    elif args.method == 'only_tracker_init':
        rgb_intrinsics = reader.get_intrinsics(all_sids[0], reader.map_sids_vids[all_sids[0]][0])
        K, height, width = intrinsics2Kres(**rgb_intrinsics)
        depth_intrinsics = rgb_intrinsics if args.use_depth else None  # same for YCBV
        tracker = Tracker(OBJ_MODEL_DIRS[ds_name], 'all', eval_cfg.tracker_cfg, rgb_intrinsics, depth_intrinsics, np.eye(4))
        tracker.init()
    
    else:
        raise ValueError(f'Method {args.method} not defined')
    ##############################################################

    if args.suffix != '':
        run_name +=  f'-{args.suffix}'


    RESULTS_DIR_NAME = Path('results')
    RESULTS_DIR_NAME.mkdir(exist_ok=True)
    EVALUATIONS_DIR_NAME = Path('evaluations')
    EVALUATIONS_DIR_NAME.mkdir(exist_ok=True)



    # bop result file name stricly formatted:<method>_<ds_name>-<split>.csv
    result_bop_eval_filename = f'{run_name}_{ds_name}-test.csv'
    result_bop_eval_path = RESULTS_DIR_NAME / result_bop_eval_filename
    print(result_bop_eval_path)
    if args.run_inference and not args.overwrite:
        if result_bop_eval_path.exists():
            out = input(f'{result_bop_eval_path} already exists, overwrite? [y]/n')
            if out == 'n':
                exit()

    if args.run_inference:
        print(f'Running {run_name}')
        time.sleep(2)

        # If Localizer class is used explicitely, init/warmup here
        if args.method in ['cosyonly', 'cosyrefined', 'trackfromstart']:
            localizer = Localizer(ds_name, eval_cfg.localizer_cfg)
            # Warmup
            sid0 = all_sids[0]
            vid0 = reader.map_sids_vids[sid0][0]
            rgb = reader.get_obs(sid0, vid0).rgb
            rgb_intrinsics = reader.get_intrinsics(sid0, vid0)
            K, height, width = intrinsics2Kres(**rgb_intrinsics)
            localizer.predict(rgb, K, n_coarse=1, n_refiner=eval_cfg.localizer_cfg.n_refiner)

        if args.method == 'threaded' and args.use_cosypose_as_tracker:
            sid0 = all_sids[0]
            vid0 = reader.map_sids_vids[sid0][0]
            rgb = reader.get_obs(sid0, vid0).rgb
            rgb_intrinsics = reader.get_intrinsics(sid0, vid0)
            continuous_tracker = ContinuousTrackerCosytrack(
                localizer_cfg=eval_cfg.localizer_cfg,
                ds_name=eval_cfg.ds_name,
                rgb_intrinsics=rgb_intrinsics,
                collect_statistics=False,
                reset_after_n=args.reset_after_n
            )

            continuous_tracker(rgb, sid0, vid0)

        for sid in all_sids:
            vids = reader.map_sids_vids[sid]

            rgb_intrinsics = reader.get_intrinsics(sid, vids[0])
            K, height, width = intrinsics2Kres(**rgb_intrinsics)
            depth_intrinsics = rgb_intrinsics if args.use_depth else None  # same for YCBV
            obs = reader.get_obs(sid, vids[0])
            rgb = obs.rgb
            depth = obs.depth if args.use_depth else None

            accepted_objects = reader.get_object_names_in_scene(sid)

            if args.method == 'threaded' and not args.use_cosypose_as_tracker:
                continuous_tracker = ContinuousTracker(
                    tracker_cfg=eval_cfg.tracker_cfg,
                    localizer_cfg=eval_cfg.localizer_cfg,
                    ds_name=eval_cfg.ds_name,
                    rgb_intrinsics=rgb_intrinsics,
                    depth_intrinsics=depth_intrinsics,
                    collect_statistics=False,
                    fake_localization_delay=args.fake_localization_delay,
                    accepted_objects=accepted_objects,
                    reset_after_n=args.reset_after_n
                )

            if args.method == 'threaded':
                # Warmup and init
                obj_poses = reader.predict_gt(sid, vids[0]) if args.use_gt_for_localization else None
                continuous_tracker(rgb, depth, obj_poses, sid, vids[0])

            elif args.method == 'cosyrefined':
                tracker = Tracker(OBJ_MODEL_DIRS[ds_name], 'all', eval_cfg.tracker_cfg, rgb_intrinsics, depth_intrinsics, np.eye(4))
                tracker.init()

            elif args.method == 'trackfromstart':
                tracker = Tracker(OBJ_MODEL_DIRS[ds_name], 'all', eval_cfg.tracker_cfg, rgb_intrinsics, depth_intrinsics, np.eye(4))
                tracker.init()
                # Init tracker poses on the first image using GT or localizer
                if args.use_gt_for_localization:
                    obj_poses = reader.predict_gt(sid, vids[0])
                else:
                    obj_poses, _ = localizer.predict(rgb, K, n_coarse=1, n_refiner=args.n_refiner)

                tracker.detected_bodies(obj_poses)

            elif args.method == 'ActorSystem':
                tp = MultiTrackerProxy()
                tp.warmup_localizer()

                if PRECHARGE:
                    req = TrackerRequest(rgb, img_id=0)
                    tp.feed_image(req=req, block=True)
                    tp._trigger_localizer_polling()

                tp.register_for_results()

            elif args.method == 'cosyonly':
                pass

            else:
                raise NotImplementedError(args.method+' not implemented')


            # Preload observation for the scene as there is a bit I/O overhead
            observations = [reader.get_obs(sid, vid) for vid in vids]

            N_views = len(vids)
            dt_method = 0.0
            for i in range(N_views):
                vid = vids[i]

                obs = observations[i]
                rgb = obs.rgb
                depth = obs.depth if args.use_depth else None

                if args.method == 'threaded':
                    t = time.perf_counter()
                    # obj_poses = reader.predict_gt(sid, vid) if args.use_gt_for_localization else None
                    obj_poses = None
                    obj_poses, scores = continuous_tracker(rgb, depth, obj_poses, sid, vid)

                    dt_method = time.perf_counter() - t
                    if reader.check_if_in_targets(sid, vid):
                        for obj_name, TCO in obj_poses.items():
                            score = scores[obj_name] if scores is not None else 1.0
                            append_result(all_bop19_results, sid, obj_name2id(obj_name), vid, score, TCO, dt_method)

                    rate.sleep()
                
                elif args.method == 'cosyrefined':
                    if reader.check_if_in_targets(sid, vid):
                        t = time.perf_counter()
                        obj_poses, scores = localizer.predict(rgb, K, n_coarse=1, n_refiner=eval_cfg.localizer_cfg.n_refiner)
                        
                        # Reset current estimations and run track
                        tracker.detected_bodies(obj_poses, scores)
                        tracker.set_image(rgb, depth)
                        tracker.track()

                        obj_poses, scores = tracker.get_current_preds()
                        dt_method = time.perf_counter() - t
                        for obj_name in obj_poses:
                            append_result(all_bop19_results, sid, obj_name2id(obj_name), vid, scores[obj_name], obj_poses[obj_name], dt_method)

                elif args.method == 'cosyonly':
                    if reader.check_if_in_targets(sid, vid):
                        obs = observations[i]
                        t = time.perf_counter()
                        obj_poses, scores = localizer.predict(rgb, K, n_coarse=1, n_refiner=eval_cfg.localizer_cfg.n_refiner)
                        dt_method = time.perf_counter() - t
                        for obj_name in obj_poses:
                            append_result(all_bop19_results, sid, obj_name2id(obj_name), vid, scores[obj_name], obj_poses[obj_name], dt_method)

                elif args.method == 'trackfromstart':
                    # First image of the scene -> init with something 
                    t = time.perf_counter()
                    tracker.set_image(rgb, depth)
                    tracker.track()
                    obj_poses, scores = tracker.get_current_preds()
                    dt_method = time.perf_counter() - t
                    scores = {obj_name: 1.0 for obj_name in obj_poses}
                    if reader.check_if_in_targets(sid, vid):
                        for obj_name, TCO in obj_poses.items():
                            score = scores[obj_name] if scores is not None else 1.0
                            append_result(all_bop19_results, sid, obj_name2id(obj_name), vid, score, TCO, dt_method)

                    tracker.update_viewers()

                elif args.method == 'ActorSystem':
                    obs = observations[i]
                    t = time.perf_counter()

                    if i == 0 and not PRECHARGE:
                        tp.feed_image(TrackerRequest(img=rgb, img_id=i), block=(i==0))
                        tp._trigger_localizer_polling()
                    else:
                        tp.feed_image(TrackerRequest(img=rgb, img_id=i), block=False)


                    while True:
                        if i > 0:
                            res = tp.listen_for_results()
                        else: 
                            res = tp.get_latest_available_estimate(wait_for_valid_res=True)
                        assert isinstance(res, TrackerRequest)
                        if res.img_id >= i:
                            break

                    obj_poses = res.poses_result

                    # scene_id, obj_id, view_id, score, TCO, dt
                    dt_method = time.perf_counter() - t
                    scores = None
                    for obj_name, TCO in obj_poses.items():
                        score = scores[obj_name] if scores is not None else 1.0
                        # print('  obj_name, score: ', obj_name, score)
                    if reader.check_if_in_targets(sid, vid):
                        for obj_name, TCO in obj_poses.items():
                            score = scores[obj_name] if scores is not None else 1.0
                            # print('  obj_name, score: ', obj_name, score)
                            append_result(all_bop19_results, sid, obj_name2id(obj_name), vid, score, TCO, dt_method)

                    rate.sleep()


                if i % args.print_info_every == 0:
                    print(f'Scene: {sid}/{all_sids[-1]}, View: {vid}/{vids[-1]}')
                    print('track + update_viewers (ms)', 1000*dt_method)

            
            if args.method == 'threaded':
                # Terminate localizer subprocess
                continuous_tracker.finish()

            # if args.method == 'ActorSystem':
            #     # Terminate localizer subprocess
            #     tp.shutdown()


    if args.run_inference:
        inout.save_bop_results(result_bop_eval_path.as_posix(), all_bop19_results)

    if args.run_evaluation:
        print(f'\nEvaluating {run_name}\n')
        run_bop_evaluation(result_bop_eval_filename, RESULTS_DIR_NAME, EVALUATIONS_DIR_NAME)

    # vid_name = f'result_{eval_cfg.ds_name}_{sid}.mp4'
    # create_video_from_images(tracker.imgs_dir, vid_name=vid_name)