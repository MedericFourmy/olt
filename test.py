import time
import yaml
import numpy as np
import quaternion
from PIL import Image
from pathlib import Path
import cv2

with open('cam_d435_640.yaml', 'r') as f:
    cam = yaml.load(f.read(), Loader=yaml.UnsafeLoader)

# example_dir = LOCAL_DATA_DIR / 'examples' / 'crackers' 
# rgb_path = example_dir / 'image_rgb.png'
rgb_path = '/home/mfourmy/Documents/ciirc_research/data/banana_video/bananas/bgr8_00079.png'
color_read_flags = cv2.IMREAD_COLOR + cv2.IMREAD_ANYDEPTH

# pyicg coherent
im_pyicg = cv2.imread(rgb_path, color_read_flags)

# CosyPose coherent
im = Image.open(rgb_path)
rgb = np.array(im, dtype=np.uint8)


###########################
###########################
########## COSY ###########
###########################
###########################

from happypose.pose_estimators.cosypose.cosypose.utils.cosypose_wrapper import CosyPoseWrapper
from happypose.toolbox.inference.types import ObservationTensor
from happypose.pose_estimators.cosypose.cosypose.config import LOCAL_DATA_DIR

def intrinsics2K(fu, fv, ppu, ppv, **not_used):
    return np.array([
        fu, 0, ppu,
        0, fv, ppv,
        0, 0, 1,
    ]).reshape((3,3))


obj_dataset = 'ycbv'
cosy_wrapper = CosyPoseWrapper(dataset_name=obj_dataset, n_workers=8)
pose_estimator = cosy_wrapper.pose_predictor
# create an observation tensor for cosypose runner


K = intrinsics2K(**cam['intrinsics_color'])


observation = ObservationTensor.from_numpy(rgb, None, K)
print(type(observation))
print(type(observation))
print(type(observation))


# labels returned by cosypose are the same as the "local_data/urdfs/ycbv" ones
# -> obj_000010 == banana
# preds = cosy_wrapper.inference(observation)  # same
preds, _ = pose_estimator.run_inference_pipeline(observation, run_detector=True, n_coarse_iterations=1, n_refiner_iterations=2)

if (isinstance(preds, list)):
    raise ValueError('NO OBJECT DETECTED IN THE IMAGE')


# TODO: extract automatically the index from preds.infos.label
pose_happy = preds.poses[0].numpy()
print('pose_happy: ', pose_happy)





###########################
###########################
########## PYICG ##########
###########################
###########################
import pyicg

def tq_to_SE3(t, q):
    """
    t: translation as list or array
    q: quaternion as list or array, expected order: xyzw
    out: 4x4 array representing the SE(3) transformation
    """
    T = np.eye(4)
    T[:3,3] = t
    # np.quaternion constructor uses wxyz order convention
    quat = np.quaternion(q[3], q[0], q[1], q[2]).normalized()
    T[:3,:3] = quaternion.as_rotation_matrix(quat)
    return T


tracker = pyicg.Tracker('tracker', synchronize_cameras=False)

renderer_geometry = pyicg.RendererGeometry('renderer geometry')
color_camera = pyicg.DummyColorCamera('cam_color')
color_camera.color2depth_pose = tq_to_SE3(cam['trans_d_c'], cam['quat_d_c_xyzw'])
color_camera.intrinsics = pyicg.Intrinsics(**cam['intrinsics_color'])

# Viewers
color_viewer = pyicg.NormalColorViewer('color_viewer', color_camera, renderer_geometry)
# color_viewer.StartSavingImages('tmp', 'png')
color_viewer.set_opacity(0.5)  # [0.0-1.0]
tracker.AddViewer(color_viewer)

# Renderers (preprocessing)
# color_depth_renderer = pyicg.FocusedBasicDepthRenderer('color_depth_renderer', renderer_geometry, color_camera)

# TODO: FOR LOOP!
# Bodies
objects_dir = Path('/home/mfourmy/Documents/ciirc_research/data/local_data_happypose/urdfs/ycbv')
object_files = {}
for obj_dir in objects_dir.iterdir():
    obj_files = list(obj_dir.glob('*.obj'))
    if len(obj_files) == 1:
        obj_path: Path = obj_files[0]
        object_files[obj_dir.name] = obj_path
    else:
        print('PROBLEM')

accepted_objs = {
    'obj_000010',
    'obj_000016',
    }
# geometry_unit_in_meter_ycbv_urdf = 0.001
geometry_unit_in_meter_ycbv_urdf = 0.001
body_map = {
    obj_name: pyicg.Body(
        name=obj_name,
        geometry_path=obj_path.as_posix(),
        geometry_unit_in_meter=geometry_unit_in_meter_ycbv_urdf,
        geometry_counterclockwise=True,
        geometry_enable_culling=True,
        geometry2body_pose=np.eye(4),
        silhouette_id=0
    )
    for obj_name, obj_path in object_files.items()
    if obj_name in accepted_objs
}



# Models
tmp_dir = Path('.')
body_name = 'obj_000010'
body_banana = body_map[body_name]
region_model_path = tmp_dir / (body_name + '_region_model.bin')
region_model = pyicg.RegionModel(body_name + '_region_model', body_banana, region_model_path.as_posix())

renderer_geometry.AddBody(body_map[body_name])
# color_depth_renderer.AddReferencedBody(body_map[body_name])


# Modalities
region_modality = pyicg.RegionModality(body_name + '_region_modality', body_banana, color_camera, region_model)

optimizer = pyicg.Optimizer(body_name+'_optimizer')
optimizer.AddModality(region_modality)
tracker.AddOptimizer(optimizer)

# Do all the necessary heavy preprocessing
ok = tracker.SetUp()
if not ok:
    raise(ValueError('Error in SetUp'))


print(tracker.n_corr_iterations)
print(tracker.n_update_iterations)
# tracker.n_update_iterations = 2
tracker.n_update_iterations = 5


# color_camera.image = im_pyicg
color_camera.image = rgb
ok = tracker.UpdateCameras(True)  # poststep verifying the images have been properly setup
if not ok:
    raise ValueError('Something is wrong with the provided images')

body_banana.body2world_pose = pose_happy

it = 0
tracker.StartModalities(it)

t = time.time()
tracker.ExecuteTrackingCycle(it)
print('ExecuteTrackingCycle (ms)', 1000*(time.time() - t))

tracker.UpdateViewers(it)

# cv2.waitKey(0)