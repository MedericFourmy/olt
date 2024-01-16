import cv2
import numpy as np
import pickle
import json
from pathlib import Path


#################
# Reading dataset
#################
DS_NAME = 'ycbv'
ds_dir = Path('/home/mfourmy/Documents/ciirc_research/data/blender_scenes')

cam_path = ds_dir / 'camera_data.json'
frames_dir = ds_dir / 'frames'
gt_path = ds_dir / 'frames_gt.p'

cam_params = json.loads(cam_path.read_text())
with gt_path.open('rb') as f:
    gt_lst = pickle.load(f)
img_paths = sorted(list(frames_dir.glob('*.png')))

K, (h,w) = np.array(cam_params['K']), cam_params['resolution']

label_map = {
    '02_cracker_box': 'obj_000002', 
    '03_sugar_box': 'obj_000003',
    '07_pudding_box': 'obj_000007',
    '12_bleach_cleanser': 'obj_000012',
}


T_wc_gt_lst = [gt['Camera'] for gt in gt_lst]
# !! assume 1 object of each label in every frame
T_co_gt_dict = [
    {
        label_map[olabel]: gt[olabel] for olabel in label_map
    }
    for gt in gt_lst
]
T_co0_gt_dic = T_co_gt_dict[0]  # GT
T_wo_gt_dic = {k: T_wc_gt_lst[0] @ T_co0_gt_dic[k] for k in T_co0_gt_dic}  # GT


color_read_flags = cv2.IMREAD_COLOR + cv2.IMREAD_ANYDEPTH
img_lst = [cv2.imread(im_path.as_posix(), color_read_flags) for im_path in img_paths]  # loads a dtype=uint8 array


#################
# Creating tracker + localizer
#################
from olt.tracker import Tracker
from olt.localizer import Localizer
from olt.config import OBJ_MODEL_DIRS, TrackerConfig, LocalizerConfig, CameraConfig
from olt.utils import tq_to_SE3, Kres2intrinsics


TEST_IMG_IDS = [2]   # One view -> high depth uncertainty, depth under estimated
TEST_IMG_IDS = [2,3]   # Two similar view -> slightly lower uncertainty, depth still under estimated
TEST_IMG_IDS = [2,10]   # Two views with higher baseline -> crushed uncertainty, good est
TEST_IMG_IDS = [2,10,15,30]   # Multiviews -> crushed uncertainty, good estimation
imgs = [cv2.cvtColor(img_lst[img_idx], cv2.COLOR_BGR2RGB) for img_idx in TEST_IMG_IDS] 

# Use 1rst frame to localize objects
localizer_cfg = LocalizerConfig() 
localizer_cfg.n_workers = 1
localizer_cfg.n_refiner = 1
localizer_cfg.detector_threshold = 0.6
localizer = Localizer(DS_NAME, localizer_cfg)
T_co0_dic, scores = localizer.predict(imgs[0], K, n_coarse=1, n_refiner=1)
T_wo0_dic = {k: T_wc_gt_lst[TEST_IMG_IDS[0]] @ T_co0_dic[k] for k in T_co0_dic}

# Refine pose using multiple views
accepted_objs = list(label_map.values())
tracker_cfg = TrackerConfig() 
tracker_cfg.viewer_display = True
tracker_cfg.viewer_save = True

camera_cfgs = [
    CameraConfig(
        rgb_intrinsics=Kres2intrinsics(K,w,h),
        color2world_pose=T_wc_gt_lst[img_idx]
    )
    for img_idx in TEST_IMG_IDS
]

tracker = Tracker(OBJ_MODEL_DIRS[DS_NAME], accepted_objs, tracker_cfg, camera_cfgs)
tracker.init()
# tracker.tracker.n_corr_iterations = 100
# tracker.tracker.n_update_iterations = 100
tracker.set_images(imgs)
tracker.detected_bodies(T_wo0_dic)
dt = tracker.track()
# dt = tracker.track()
# print('track took (ms):', 1000*dt)
tracker.update_viewers()
# cv2.waitKey(0)

T_wo_dic, scores = tracker.get_current_preds()


# Expe 1:
# one object, display 

import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt


def visualize_poses(T_wc_lst, T_wo_dic, Q_wo=None, fig_name=None):
    fig = plt.figure(fig_name)
    axes = fig.add_subplot(projection='3d')
    plt.cla()

    axis_length = 0.1
    for T_wc in T_wc_lst:
        gtsam_plot.plot_pose3(fig.number, gtsam.Pose3(T_wc), 2*axis_length)
    if Q_wo is None:
        Q_wo = {label: None for label in T_wo_dic}
    for label in T_wo_dic:
        gtsam_plot.plot_pose3(fig.number, gtsam.Pose3(T_wo_dic[label]), axis_length, Q_wo[label])

    xyz_min = -1 
    xyz_max =  1
    axes.set_xlim3d(xyz_min, xyz_max)
    axes.set_ylim3d(xyz_min, xyz_max)
    axes.set_zlim3d(xyz_min, xyz_max)
    
    return fig


sig = 0.05
Q = sig**2 * np.eye(6)
# Q_preds = {label: Q for label in T_co_dic}

Q_co_preds = {label: 300*np.linalg.inv(tracker.links[label].hessian()) for label in tracker.links}

T_wc_gt_imgs = [T_wc_gt_lst[img_idx] for img_idx in TEST_IMG_IDS]
visualize_poses(T_wc_gt_imgs, T_wo_dic, Q_co_preds, fig_name='Preds')
visualize_poses(T_wc_gt_imgs, T_wo_gt_dic, fig_name='GT')

plt.show()

