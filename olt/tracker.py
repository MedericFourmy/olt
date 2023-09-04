"""

Vocabulary
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



from olt.utils import tq_to_SE3, obj_name2id
from olt.config import TrackerConfig


class Tracker:
    """
    params:
    - obj_model_dir: path to directory containing <model_name>.obj files
                     !! BOP datasets contain .ply objects, not .obj, we need to find them in another folder
    - tmp_dir: directory where cached computations are stored (e.g. RegionModel files)
    - accepted_objs: determines which object to load ('all' str or list of <model_name> str)
    - rgb_intrinsics: rgb camera intrinsics as dict with keys: fu, fv, ppu, ppv, width, height
    - depth_intrinsics: depth camera rgb_intrinsics as dict with keys: fu, fv, ppu, ppv, width, height
    """
    def __init__(self, 
                 obj_model_dir: Union[str,Path],
                 accepted_objs: Union[set[str],str],
                 cfg: TrackerConfig,
                 rgb_intrinsics: dict,
                 depth_intrinsics: Union[dict,None] = None,
                 color2depth_pose: Union[np.ndarray,None] = None,
                 ) -> None:
        
        if cfg.use_depth:
            assert depth_intrinsics is not None

        if cfg.measure_occlusions:
            assert cfg.use_depth

        # print('TrackerConfig:\n', cfg)
        self.cfg = cfg
        self.tmp_dir = Path(self.cfg.tmp_dir_name)
        self.obj_model_dir = Path(obj_model_dir)
        self.accepted_objs = accepted_objs

        # Camera parameters
        self.rgb_intrinsics = rgb_intrinsics
        self.depth_intrinsics = depth_intrinsics
        self.color2depth_pose = color2depth_pose

        # some other parameters
        self.geometry_unit_in_meter_ycbv_urdf = 0.001

        self.active_tracks = []

        self.image_set = False

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
        self.color_camera.color2depth_pose = self.color2depth_pose
        self.color_camera.camera2world_pose = np.eye(4)  # color camera fixed at the origin of the world
        self.color_camera.intrinsics = pyicg.Intrinsics(**self.rgb_intrinsics)

        if self.cfg.use_depth:
            self.depth_camera = pyicg.DummyDepthCamera('cam_depth')
            self.depth_camera.color2depth_pose = self.color2depth_pose
            self.depth_camera.camera2world_pose = self.color_camera.depth2color_pose  # world shifted by depth2color transformation
            self.depth_camera.intrinsics = pyicg.Intrinsics(**self.depth_intrinsics)

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
        self.depth_models = {}
        self.depth_modalities = {} 
        self.optimizers = {}
        for bname, body in self.bodies.items():
            region_model_path = self.tmp_dir / (bname + '_region_model.bin')
            self.region_models[bname] = pyicg.RegionModel(bname + '_region_model', body, region_model_path.as_posix())
            if self.cfg.use_depth:
                depth_model_path = self.tmp_dir / (bname + '_depth_model.bin')
                self.depth_models[bname] = pyicg.DepthModel(bname + '_depth_model', body, depth_model_path.as_posix())

            # Q: Possible to create on the fly?
            self.region_modalities[bname] = pyicg.RegionModality(bname + '_region_modality', body, self.color_camera, self.region_models[bname])
            if self.cfg.use_depth:
                self.depth_modalities[bname] = pyicg.DepthModality(bname + '_depth_modality', body, self.depth_camera, self.depth_models[bname])

                # Detect occlusion using measured depth and use it to improve estimates
                if self.cfg.measure_occlusions:
                    self.region_modalities[bname].MeasureOcclusions(self.depth_camera)
                    self.depth_modalities[bname].MeasureOcclusions()

            # TODO: Add only the tracked body to the viewer?
            self.renderer_geometry.AddBody(body)

            # Add remove optimizers?
            self.optimizers[bname] = pyicg.Optimizer(bname+'_optimizer', self.cfg.tikhonov_parameter_rotation, self.cfg.tikhonov_parameter_translation)

            # Modalities
            self.optimizers[bname].AddModality(self.region_modalities[bname])
            if self.cfg.use_depth:
                self.optimizers[bname].AddModality(self.depth_modalities[bname])

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
        # obj_name: 'obj_'
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
        {
            'obj_000010': pose1,
            'obj_000016': pose2,
            'obj_000010': pose3,
            ...
        }

        FOR NOW: assume unique objects and reinitialize there pose
        """

        # Implementation 1: just update the current estimates, 1 object per cat
        self.active_tracks = []
        for obj_name, T_co in detections.items():
            self.active_tracks.append(obj_name)
            if isinstance(T_co, pyicg.Body):
                self.bodies[obj_name].body2world_pose = T_co.body2world_pose
            elif isinstance(T_co, np.ndarray):
                assert T_co.shape == (4,4)
                self.bodies[obj_name].body2world_pose = T_co
            else:
                raise ValueError(f"Expected homogenous transformation or pyicg bodies, but got {type(T_co)}")

        # Implementation 2: matching (if multiple instances of same object)
        
    def update_K(self, K, width, height):
        """
        Some objects of pyicg do not allow update of K without some modifications (e.g. RendererGeometry).
        """
        raise NotImplementedError('update_K')

    def set_image(self, rgb: np.array, depth: Union[np.array,None] = None):
        self.color_camera.image = rgb
        if self.cfg.use_depth:
            self.depth_camera.image = depth

        # verifying the images have been properly setup
        ok = self.tracker.UpdateCameras(True) 
        if not ok:
            raise ValueError('Something is wrong with the provided images')
        self.image_set = True

    def track(self):
        assert self.image_set, "No image was set yet! Call tracker.set_image(img)"
        if self.iteration == 0:
            print('StartModalities!')
            self.tracker.StartModalities(self.iteration)

        t = time.time()
        self.tracker.ExecuteTrackingCycle(self.iteration)
        dt = time.time() - t

        self.iteration += 1

        return dt

    def update_intrinsics(self, rgb_intrinsics):
        raise NotImplementedError('update_intrinsics not implementable with current ICG version')
        self.rgb_intrinsics = rgb_intrinsics

        # camera
        self.color_camera.rgb_intrinsics = pyicg.Intrinsics(**rgb_intrinsics)

        # modalities
        for rm in self.region_modalities.values:
            rm.PrecalculateCameraVariables()  

        # viewers
        # TODO: not possible right now!

        # FocusedRenderer (for occlussions)
        # TODO: not possible right now!


    def update_viewers(self):
        t = time.time()
        self.tracker.UpdateViewers(self.iteration)
        dt = time.time() - t
        return dt

    def get_current_preds(self):
        preds = {}
        for obj_name in self.active_tracks:
            preds[obj_name] = self.bodies[obj_name].body2world_pose

        return preds


