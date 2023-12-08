        


import os
import numpy as np
import collections
from PIL import Image
from pathlib import Path
import pinocchio as pin
from matplotlib import pyplot as plt

from bop_toolkit_lib.inout import load_json, load_bop_results
from olt.config import OBJ_MODEL_DIRS, HAPPYPOSE_DATA_DIR, BOP_DS_DIRS
from olt.evaluation_tools import run_bop_evaluation, BOPDatasetReader
from olt.overlay_rendering import render_overlays
from olt.utils import obj_id2name



RUN_EVAL = True
PLOT = False
RENDER = False
DS_NAME = 'ycbv'

OBJECTS_OF_INTEREST = [15]
OBJECTS_OF_INSTEREST_NAMES = [obj_id2name(id) for id in OBJECTS_OF_INTEREST]

# 01 002_master_chef_can
# 02 003_cracker_box
# 03 004_sugar_box
# 04 005_tomato_soup_can
# 05 006_mustard_bottle
# 06 007_tuna_fish_can
# 07 008_pudding_box
# 08 009_gelatin_box
# 09 010_potted_meat_can
# 10 011_banana
# 11 019_pitcher_base
# 12 021_bleach_cleanser
# 13 024_bowl
# 14 025_mug
# 15 035_power_drill
# 16 036_wood_block
# 17 037_scissors
# 18 040_large_marker
# 19 051_large_clamp
# 20 052_extra_large_clamp
# 21 061_foam_brick


name_dict = {
    'threaded': "ours",
    "trackfromstart": "tracker",
    "ActorSystem": "ActorSystem",
    "localizerwithidentitymapastracker": "localizer with identity map as tracker",
    "threaded-synt+real-bullet-30Hz-nr2-rgb-ntrackit2-tikhohigh-mssd_ycbv-test.csv": "OLT RGB",
    "threaded-synt+real-bullet-30Hz-nr2-rgb-ntrackit0-tikhohigh-mssd_ycbv-test.csv": "OLTWIthIidentityTracker RGB",
    'trackfromstart-from-gt=False-rgbd-tikhohigh-mssd_ycbv-test.csv': "TrackerInitWithLocalizer RGB",
    "threaded-synt+real-bullet-30Hz-nr2-rgbd-ntrackit2-tikhohigh-mssd_ycbv-test.csv": "OLT RGB-D",
    'trackfromstart-from-gt=False-rgb-tikhohigh-mssd_ycbv-test.csv': "TrackerInitWithLocalizer RGB-D"}

reader = BOPDatasetReader(DS_NAME, load_depth=False)


def render_overlays_for_scene(ds_name, scene_id, T_co_lst_per_img, name_prefix=""):

    number_imgs = len(T_co_lst_per_img)

    colors_lst = [
        [(0, 255, 0)] for _ in range(number_imgs)
    ]
    
    for i in range(1, number_imgs):
        if i >= 260:
            break
        
        vid = reader.map_sids_vids[scene_id]
        obs = reader.get_obs(scene_id, vid)
        K = obs.camera_data.K
        height, width = obs.camera_data.resolution
        rgb = obs.rgb        

        # TODO: put after the for loop and pass longer lists to be faster
        render_overlays(ds_name, [rgb], K, height, width, OBJECTS_OF_INSTEREST_NAMES, 
                        [T_co_lst_per_img[i]], [colors_lst[i]], 
                        [f"{name_prefix}_{ds_name}_ds_sid_{scene_id}_vid_{i}"])


def get_target_file_string(reader: BOPDatasetReader, scene_ids):
    s = ''
    s += '['

    for sid in scene_ids:
        object_ids = reader.get_object_ids_in_scene(sid)
        view_ids = reader.map_sids_vids[sid]
        for vid in view_ids:
            for obj in object_ids:
                s += "\n"
                # Assume only 1 object of each category per scene
                s += f'{{"im_id": {vid}, "inst_count": 1, "obj_id": {obj}, "scene_id": {sid}}},'

    s = s[:-1] # remove last comma
    s += '\n]\n'
    return s


# RESULTS_FILENAMES = [
#     "threaded-synt+real-bullet-30Hz-nr2-rgb-ntrackit0-tikhohigh-mssd_ycbv-test.csv", 
#     "threaded-synt+real-bullet-30Hz-nr2-rgbd-ntrackit2-tikhohigh-mssd_ycbv-test.csv", 
#     "threaded-synt+real-bullet-30Hz-nr2-rgb-ntrackit2-tikhohigh-mssd_ycbv-test.csv",
#     "trackfromstart-from-gt=False-rgbd-tikhohigh-mssd_ycbv-test.csv",
#     "trackfromstart-from-gt=False-rgb-tikhohigh-mssd_ycbv-test.csv"
# ]


RESULTS_FILENAMES = [
    'threaded-synt+real-bullet-30Hz-nr2-rgb-ntrackit2-tikhohigh-mssd_ycbv-test.csv',
]



all_scene_ids = list(reader.map_sids_vids.key())
all_scene_ids = [all_scene_ids[2]]
targets_file = BOP_DS_DIRS[DS_NAME] / f"test_targets_all_views.json"
with open(targets_file.as_posix(), 'w') as f:
    f.write(get_target_file_string(reader, scene_ids=all_scene_ids))


RESULTS_DIR_NAME = Path('results')
EVALUATIONS_DIR_NAME = Path('evaluations_viz')
RESULTS_DIR_NAME.mkdir(exist_ok=True)


if RUN_EVAL:
    for result_bop_eval_filename in RESULTS_FILENAMES:
        result_bop_eval_path = RESULTS_DIR_NAME / result_bop_eval_filename
        run_bop_evaluation(result_bop_eval_filename, RESULTS_DIR_NAME, EVALUATIONS_DIR_NAME, targets_file.name)

if RENDER:

    for i, filename in enumerate(RESULTS_FILENAMES):
        # just for rendering the next one now
        if i != 4:
            continue

        result_bop_eval_filename = filename
        result_bop_eval_path = RESULTS_DIR_NAME / result_bop_eval_filename
        estimates = load_bop_results(os.path.abspath(result_bop_eval_path))

        T_co_lst_per_img = []

        for est in estimates:
            if est["obj_id"] in OBJECTS_OF_INTEREST:
                T_co = np.eye(4)
                T_co[:3,:3] = est['R']
                T_co[:3,3] = 0.001 * est['t'].squeeze()
                T_co_lst_per_img.append([T_co])

            
        # TODO: loop over scenes?
        render_overlays_for_scene(DS_NAME, all_scene_ids[0], T_co_lst_per_img, name_prefix=name_dict[filename].replace(" ", "-"))

        print("still here")
        # run_bop_evaluation(result_bop_eval_filename, RESULTS_DIR_NAME, EVALUATIONS_DIR_NAME, os.path.basename(targets_file))



def plot_error_over_time(paths, names, objects=None):
    from bop_toolkit_lib.inout import load_json
    # path = "evaluations/threaded-synt+real-bullet-15Hz-rgb-scene50_ycbv-test/error=mspd_ntop=-1/errors_000050.json"
    # path = "evaluations/ActorSystem-synt+real-bullet-15Hz-rgb-scene50_ycbv-test/error=mspd_ntop=-1/errors_000050.json"
    # path = "evaluations/trackfromstart-from-gt=False-rgb-scene50_ycbv-test/error=mspd_ntop=-1/errors_000050.json"
    # path = "evaluations/threaded-synt+real-bullet-15Hz-rgb-scene50_ycbv-test/error=mssd_ntop=-1/errors_000050.json"
    # path = "evaluations/trackfromstart-from-gt=False-rgb-scene50_ycbv-test/error=mssd_ntop=-1/errors_000050.json"
    # path = "evaluations/ActorSystem-synt+real-bullet-15Hz-rgb-scene50_ycbv-test/error=mssd_ntop=-1/errors_000050.json"

    fig, ax = plt.subplots()


    for name, path in zip(names, paths):

        errs = load_json(path=path)

        # get all obj ids
        obj_ids = set()
        if objects is None:
            for err in errs:
                obj_ids.add(err["obj_id"])
        elif isinstance(objects, collections.abc.Iterable):
            obj_ids.update(objects)
    
        plot_errs = {}
        
        for err in errs:
            if not err["obj_id"] in plot_errs.keys():
                plot_errs[err["obj_id"]] = {"im_id": [], "err": []}

            plot_errs[err["obj_id"]]["im_id"].append(err["im_id"])
            plot_errs[err["obj_id"]]["err"].append(list(err["errors"].values())[0][0])

        for obj_id, data in plot_errs.items():
            if obj_id not in obj_ids:
                continue
            ax.plot(data["im_id"], data["err"], label=f"{obj_id} {name}", marker='x')

    fig.legend()
    # fig.show()

    fig.savefig(f"err_over_time.pdf")

    fig.show()


    print("dev plotting")



if PLOT:
    error_file_paths = []
    names = []
    
    
    for filename in RESULTS_FILENAMES:
        result_bop_eval_filename = filename 
        result_bop_eval_path = RESULTS_DIR_NAME / result_bop_eval_filename
        # run_bop_evaluation(result_bop_eval_filename, RESULTS_DIR_NAME, EVALUATIONS_DIR_NAME, os.path.basename(targets_file))
        

        # OLT RGB:        threaded-synt+real-bullet-30Hz-nr2-rgb-ntrackit2-tikhohigh-mssd_ycbv-test.csv
        # OLTWIthIidentityTracker RGB:      threaded-synt+real-bullet-30Hz-nr2-rgb-ntrackit0-tikhohigh-mssd_ycbv-test.csv
        # TrackerInitWithLocalizer RGB:     'trackfromstart-from-gt=False-rgbd-tikhohigh-mssd_ycbv-test.csv'
        # OLT RGB-D:                        threaded-synt+real-bullet-30Hz-nr2-rgbd-ntrackit2-tikhohigh-mssd_ycbv-test.csv
        # TrackerInitWithLocalizer RGB-D: 'trackfromstart-from-gt=False-rgb-tikhohigh-mssd_ycbv-test.csv'

        


        error_file_paths.append(EVALUATIONS_DIR_NAME / filename.removesuffix(".csv") / f"error=mssd_ntop=-1" / f"errors_{scene_id:0>6}.json")
        # if "rgbd" in filename.split("-"):
        #     names.append(f'{name_dict[filename.split("-")[0]]} rgbd')
        # elif "rgb" in filename.split("-"):
        #     if "ntrackit0" in filename.split("-"):
        #         names.append(f'{name_dict[filename.split("-")[0]]} rgb')

        #     names.append(f'{name_dict[filename.split("-")[0]]} rgb')
        # else:
        #     raise ValueError(f"can't handle filename {filename}")

        names.append(name_dict[filename])

    plot_error_over_time(paths = error_file_paths, names=names, objects=OBJECTS_OF_INTEREST)

