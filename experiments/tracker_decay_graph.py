        


import collections
import os
# from olt.evaluation_tools import run_bop_evaluation
from pathlib import Path
import subprocess
import bop_toolkit_lib
from bop_toolkit_lib.inout import load_im, load_json, load_bop_results
from matplotlib import pyplot as plt

from olt.config import OBJ_MODEL_DIRS, MEGAPOSE_DATA_DIR
from PIL import Image
import numpy as np

import pinocchio as pin
from olt.overlay_rendering import render_overlays



RUN_EVAL = False
PLOT = False
RENDER = True
visib_gt_min = -1

OBJECTS_OF_INTEREST = [15]

object_labels_lst = [
    ['obj_000015']
]

T_1_10 = pin.XYZQUATToSE3([0.1, 0.05, 0.65,    -0.5, -0.5, 0.5, -0.5 ])

name_dict = {'threaded': "ours",
                 "trackfromstart": "tracker",
                 "ActorSystem": "ActorSystem",
                 "localizerwithidentitymapastracker": "localizer with identity map as tracker",
                 "threaded-synt+real-bullet-30Hz-nr2-rgb-ntrackit2-tikhohigh-mssd_ycbv-test.csv": "OLT RGB",
                 "threaded-synt+real-bullet-30Hz-nr2-rgb-ntrackit0-tikhohigh-mssd_ycbv-test.csv": "OLTWIthIidentityTracker RGB",
                 'trackfromstart-from-gt=False-rgbd-tikhohigh-mssd_ycbv-test.csv': "TrackerInitWithLocalizer RGB",
                 "threaded-synt+real-bullet-30Hz-nr2-rgbd-ntrackit2-tikhohigh-mssd_ycbv-test.csv": "OLT RGB-D",
                 'trackfromstart-from-gt=False-rgb-tikhohigh-mssd_ycbv-test.csv': "TrackerInitWithLocalizer RGB-D"}



def load_img(SCENE_ID=50, VIEW_ID=1):
    DS_NAME = 'ycbv'
    SPLIT = "test"
    # SCENE_ID = 48
    # VIEW_ID = 1

    scene_id_str = f'{SCENE_ID:06}'
    view_id_str = f'{VIEW_ID:06}'
    rgb_full_path = MEGAPOSE_DATA_DIR / f'bop_datasets/{DS_NAME}/{SPLIT}/{scene_id_str}/rgb/{view_id_str}.png'
    scene_cam_full_path = MEGAPOSE_DATA_DIR / f'bop_datasets/{DS_NAME}/{SPLIT}/{scene_id_str}/scene_camera.json'
    d_scene_camera = load_json(scene_cam_full_path)
    K = d_scene_camera[str(VIEW_ID)]['cam_K']
    K = np.array(K).reshape((3,3))

    im = Image.open(rgb_full_path)
    rgb = np.array(im, dtype=np.uint8)
    return ds_name, [rgb], K, rgb.shape[0], rgb.shape[1], 


def render_overlays_for_scene(scene_id, T_co_lst_per_img, name_prefix=""):

    number_imgs = T_co_lst_per_img.__len__()

    colors_lst = [
        [(0, 255, 0)] for _ in range(number_imgs)
    ]
    
    for i in range(1, number_imgs):
        if i >= 260:
            break
        ds_name, rgb_lst, K, height, width  = load_img(scene_id, i)

        render_overlays(ds_name, rgb_lst, K, height, width, object_labels_lst, [T_co_lst_per_img[i]], [colors_lst[i]], [f"{name_prefix}_{ds_name}_ds_sid_{scene_id}_vid_{i}"])



def run_bop_evaluation(filename, results_dir_name, evaluations_dir_name, targets_file_name):
    myenv = os.environ.copy()

    BOP_TOOLKIT_DIR = Path(bop_toolkit_lib.__file__).parent.parent
    POSE_EVAL_SCRIPT_PATH = BOP_TOOLKIT_DIR / "scripts/eval_bop19_pose.py"

    # Put results in current directory
    root_dir = Path(os.getcwd())
    results_path = root_dir / results_dir_name
    eval_path = root_dir / evaluations_dir_name

    renderer_type = 'vispy'  # other options: 'cpp', 'python'
    cmd = [
        "python",
        str(POSE_EVAL_SCRIPT_PATH),
        "--result_filenames",
        filename,
        "--results_path",
        results_path,
        "--renderer_type",
        renderer_type,
        "--eval_path",
        eval_path,
        '--targets_filename',
        targets_file_name,
        '--visib_gt_min',
        str(visib_gt_min)
    ]
    # subprocess.call(cmd, env=myenv, cwd=BOP_TOOLKIT_DIR.as_posix())
    subprocess.call(cmd, env=myenv, cwd=os.getcwd())



def write_targets_file(img_ids, objs, scene_id):
    s = ''
    s += '['
    for img_id in img_ids:
        for obj in objs:
            s += "\n"
            s += f'{{"im_id": {img_id}, "inst_count": 1, "obj_id": {obj}, "scene_id": {scene_id}}},'

    s = s[:-1] # remove last comma
    s += '\n]\n'
    return s

# filenames = ["trackfromstart-from-gt=False-rgb-scene50_ycbv-test.csv", 
#              "ActorSystem-synt+real-bullet-15Hz-rgb-scene50_ycbv-test.csv", 
#              "threaded-synt+real-bullet-15Hz-rgb-scene50_ycbv-test.csv"]
filenames = ["threaded-synt+real-bullet-30Hz-nr2-rgbd-ntrackit2-tikhohigh-mssd_ycbv-test.csv", 
             "threaded-synt+real-bullet-30Hz-nr2-rgb-ntrackit0-tikhohigh-mssd_ycbv-test.csv", 
             "threaded-synt+real-bullet-30Hz-nr2-rgb-ntrackit2-tikhohigh-mssd_ycbv-test.csv",
             "trackfromstart-from-gt=False-rgbd-tikhohigh-mssd_ycbv-test.csv",
             "trackfromstart-from-gt=False-rgb-tikhohigh-mssd_ycbv-test.csv"]


filename = filenames[1]

scene_id = 50

scene_objects = {50: [2,4,5,10,15]}
scene_len = {50: 1916}

targets_file = f"/home/behrejan/local_data_happypose/bop_datasets/ycbv/test_targets_all_scene_{scene_id}.json"
with open(targets_file, 'w') as f:
    f.write(write_targets_file(range(1,scene_len[scene_id] + 1), scene_objects[scene_id], scene_id))

ds_name = 'ycbv'


RESULTS_DIR_NAME = Path('results')
EVALUATIONS_DIR_NAME = Path('evaluations')
RESULTS_DIR_NAME.mkdir(exist_ok=True)


if RUN_EVAL:
    for filename in filenames:
        result_bop_eval_filename = filename 
        result_bop_eval_path = RESULTS_DIR_NAME / result_bop_eval_filename
        run_bop_evaluation(result_bop_eval_filename, RESULTS_DIR_NAME, EVALUATIONS_DIR_NAME, os.path.basename(targets_file))

if RENDER:

    for i, filename in enumerate(filenames):
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

            

        render_overlays_for_scene(scene_id, T_co_lst_per_img, name_prefix=name_dict[filename].replace(" ", "-"))

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
    
    
    for filename in filenames:
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

