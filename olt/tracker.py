"""

Vocabulary
- object id: id of an object in one of the datasets, e.g. obj_000010 = banana object for ycbv
- body name: name of a specific instance of a object in pyicg

Notations
- T_co: SE(3) transformation that translate and rotate a vector from frame o to frame c: c_v = T_co @ o_v
"""



import time
import numpy as np
from pathlib import Path
import shutil
from typing import Union
import pyicg



from olt.utils import tq_to_SE3
from olt.config import TrackerConfig



def detection_id_conversion(det_id: str, ds_name: str):
    if ds_name == 'ycbv':
        # For ycbv+CosyPose, labels from detection are look like 'ycbv-obj_000010'
        return det_id.split('-')[1]
    else:
        raise ValueError(f'Unknown dataset name {ds_name}')


class Tracker:
    """
    params:
    - intrinsics: camera intrinsics as dict with keys: fu, fv, ppu, ppv, width, height
    - obj_model_dir: path to directory containing <model_name>.obj files
                     !! BOP datasets contain .ply objects, not .obj, we need to find them in another folder
    - tmp_dir: directory where cached computations are stored (e.g. RegionModel files)
    - accepted_objs: determines which object to load ('all' str or list of <model_name> str)
    """
    def __init__(self, 
                 intrinsics: dict,
                 obj_model_dir: Union[str,Path],
                 accepted_objs: Union[set[str],str],
                 cfg: TrackerConfig
                 ) -> None:
        print('TrackerConfig:\n', cfg)
        self.cfg = cfg
        self.tmp_dir = Path(self.cfg.tmp_dir_name)
        self.obj_model_dir = Path(obj_model_dir)
        self.accepted_objs = accepted_objs

        # Camera parameters
        self.intrinsics = intrinsics

        # some other parameters
        self.geometry_unit_in_meter_ycbv_urdf = 0.001

    def init(self):
        # Check if paths exist
        if not self.tmp_dir.exists(): self.tmp_dir.mkdir(parents=True)
        self.imgs_dir = self.tmp_dir / 'imgs'
        # erase image directory
        if self.imgs_dir.exists(): 
            shutil.rmtree(self.imgs_dir.as_posix(), ignore_errors=True)
        self.imgs_dir.mkdir(parents=True)
        assert(self.obj_model_dir.exists())

        # Renderer for preprocessing
        self.renderer_geometry = pyicg.RendererGeometry('renderer geometry')

        self.color_camera = pyicg.DummyColorCamera('cam_color')
        # TODO: new attributes
        trans_d_c = [0,0,0]
        quat_d_c_xyzw = [0,0,0,1]
        self.color_camera.color2depth_pose = tq_to_SE3(trans_d_c, quat_d_c_xyzw)
        self.color_camera.intrinsics = pyicg.Intrinsics(**self.intrinsics)
        self.color_camera.camera2world_pose = np.eye(4)  # assume camera is fixed and aligned with world frame

        # Viewers
        color_viewer = pyicg.NormalColorViewer(self.cfg.viewer_name, self.color_camera, self.renderer_geometry)
        if self.cfg.viewer_save:
            color_viewer.StartSavingImages(self.imgs_dir.as_posix(), 'png')
        color_viewer.set_opacity(0.5)  # [0.0-1.0]
        color_viewer.display_images = self.cfg.viewer_display

        # Main class
        self.tracker = pyicg.Tracker('tracker', synchronize_cameras=False)
        self.tracker.AddViewer(color_viewer)

        # bodies: create 1 for each object model with body names = body ids
        self.bodies, self.object_files = self.create_bodies(self.obj_model_dir, self.accepted_objs, self.geometry_unit_in_meter_ycbv_urdf)

        self.region_models = {}
        self.region_modalities = {} 
        self.optimizers = {}
        for bname, body in self.bodies.items():
            region_model_path = self.tmp_dir / (bname + '_region_model.bin')
            self.region_models[bname] = pyicg.RegionModel(bname + '_region_model', body, region_model_path.as_posix())

            # Q: Possible to create on the fly?
            self.region_modalities[bname] = pyicg.RegionModality(bname + '_region_modality', body, self.color_camera, self.region_models[bname])

            # TODO: Add only the tracked body to the viewer?
            self.renderer_geometry.AddBody(body)

            # Add remove optimizers?
            self.optimizers[bname] = pyicg.Optimizer(bname+'_optimizer')

            # Modalities
            self.optimizers[bname].AddModality(self.region_modalities[bname])
            self.tracker.AddOptimizer(self.optimizers[bname])


        self.tracker.n_corr_iterations = self.cfg.n_corr_iterations
        self.tracker.n_update_iterations = self.cfg.n_update_iterations
        print('n_corr_iterations: ', self.tracker.n_corr_iterations)
        print('n_update_iterations: ', self.tracker.n_update_iterations)

        print('SETUP TRACKER')
        ok = self.tracker.SetUp()
        if not ok:
            raise(ValueError('Error in SetUp'))
        print('TRACKER SETUP OK')

        self.iteration = 0

    def create_bodies(self, 
                    object_model_dir: Path, 
                    accepted_objs: Union[set[str],str], 
                    geometry_unit_in_meter: float) -> dict[str, pyicg.Body]:
        
        # Bodies
        object_files = {}
        for obj_dir in object_model_dir.iterdir():
            obj_files = list(obj_dir.glob('*.obj'))
            if len(obj_files) == 1:
                obj_path = obj_files[0]
                object_files[obj_dir.name] = obj_path
            else:
                print('PROBLEM: less or more than one file were found')
        
        print('accepted_objs: ', accepted_objs)
        bodies = {
            obj_name: pyicg.Body(
                name=obj_name,
                geometry_path=obj_path.as_posix(),
                geometry_unit_in_meter=geometry_unit_in_meter,
                geometry_counterclockwise=True,
                geometry_enable_culling=True,
                geometry2body_pose=np.eye(4),
                silhouette_id=0
            )
            for obj_name, obj_path in object_files.items()
            if accepted_objs == 'all' or obj_name in accepted_objs
        }

        ###########################
        ## TEST BODY INIT
        ###########################
        T_bc = np.eye(4)
        # T_bc[2,3] = -0.8 
        T_bc[2,3] = -1000000000000
        for body in bodies.values():
        #     T[0,3] = np.random.random() - 0.25
            body.body2world_pose = T_bc
        ###########################
        ###########################

        return bodies, object_files

    def detected_bodies(self, detections: dict[str, np.array]):
        """
        detections: list of object id,pose pairs coming from a pose estimator like happy pose
        Multiple objects of the sane
        [
            (obj_000010, pose1),
            (obj_000016, pose2),
            (obj_000010, pose3),
            ...
        ]

        FOR NOW: assume unique objects and reinitialize there pose
        """


        # Implementation 1: just update the current e
        # stimates, 1 object per cat
        for det_id, T_co in detections.items():
            obj_id = detection_id_conversion(det_id, 'ycbv')
            self.bodies[obj_id].body2world_pose = T_co

        # Implementation 2: matching (if multiple instances of same object)
        
    def update_K(self, K, width, height):
        """
        Some objects of pyicg do not allow update of K without some modifications (e.g. RendererGeometry).
        """
        raise NotImplementedError('update_K')

    def set_image(self, img: np.array):
        self.color_camera.image = img
        # verifying the images have been properly setup
        ok = self.tracker.UpdateCameras(True) 
        if not ok:
            raise ValueError('Something is wrong with the provided images')

    def track(self):
        if self.iteration == 0:
            print('StartModalities!')
            self.tracker.StartModalities(self.iteration)

        t = time.time()
        self.tracker.ExecuteTrackingCycle(self.iteration)
        print('ExecuteTrackingCycle (ms)', 1000*(time.time() - t))

        self.iteration += 1

    def update_intrinsics(self, intrinsics):
        self.intrinsics = intrinsics

        # camera
        self.color_camera.intrinsics = pyicg.Intrinsics(**intrinsics)

        # modalities
        for rm in self.region_modalities.values:
            rm.PrecalculateCameraVariables()  

        # viewers
        # TODO: not possible right now!

        # FocusedRenderer (for occlussions)
        # TODO: not possible right now!


    def update_viewers(self):
        self.tracker.UpdateViewers(self.iteration)

