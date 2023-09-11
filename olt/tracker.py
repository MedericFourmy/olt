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

        # print('TrackerConfig:\n', cfg)
        self.cfg = cfg
        self.tmp_dir = Path(self.cfg.tmp_dir_name)
        self.obj_model_dir = Path(obj_model_dir)
        self.accepted_objs = accepted_objs

        # Camera parameters
        self.rgb_intrinsics = rgb_intrinsics
        self.depth_intrinsics = depth_intrinsics
        self.color2depth_pose = color2depth_pose if color2depth_pose is not None else np.eye(4)

        # some other parameters
        self.geometry_unit_in_meter_ycbv_urdf = 0.001

        self.active_tracks = {}

        self.image_set = False

    def init(self):
        # Check if paths exist
        if not self.tmp_dir.exists(): self.tmp_dir.mkdir(parents=True)
        self.imgs_dir = self.tmp_dir / 'imgs'
        # erase image directory
        if self.imgs_dir.exists(): 
            shutil.rmtree(self.imgs_dir.as_posix(), ignore_errors=True)
        self.imgs_dir.mkdir(parents=True, exist_ok=True)
        assert(self.obj_model_dir.exists())


        # Main class
        self.tracker = pyicg.Tracker('tracker', synchronize_cameras=False)
        
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

        if self.cfg.viewer_display or self.cfg.viewer_save:
            #########################################
            # Viewers
            self.color_viewer = pyicg.NormalColorViewer('color_'+self.cfg.viewer_name, self.color_camera, self.renderer_geometry)
            if self.cfg.viewer_save:
                self.color_viewer.StartSavingImages(self.imgs_dir.as_posix(), 'png')
            self.color_viewer.set_opacity(0.5)  # [0.0-1.0]
            self.color_viewer.display_images = self.cfg.viewer_display

            if self.cfg.use_depth:
                depth_viewer = pyicg.NormalDepthViewer('depth_'+self.cfg.viewer_name, self.depth_camera, self.renderer_geometry)
                if self.cfg.viewer_save:
                    depth_viewer.StartSavingImages(self.imgs_dir.as_posix(), 'png')
                depth_viewer.display_images = self.cfg.viewer_display

            self.tracker.AddViewer(self.color_viewer)
            display_depth = True
            if self.cfg.use_depth and display_depth:
                self.tracker.AddViewer(depth_viewer)
            #########################################

        # bodies: create 1 for each object model with body names = body ids
        self.bodies, self.object_files = self.create_bodies(self.obj_model_dir, self.accepted_objs, self.geometry_unit_in_meter_ycbv_urdf)
        self.set_all_bodies_behind_camera()

        self.region_models = {}
        self.region_modalities = {} 
        self.depth_models = {}
        self.depth_modalities = {} 
        self.optimizers = {}
        for bname, body in self.bodies.items():
            region_model_path = self.tmp_dir / (bname + '_region_model.bin')
            rm = self.cfg.region_model
            self.region_models[bname] = pyicg.RegionModel(bname + '_region_model', body, region_model_path.as_posix(),
                                                          sphere_radius=rm.sphere_radius, 
                                                          n_divides=rm.n_divides, 
                                                          n_points=rm.n_points, 
                                                          max_radius_depth_offset=rm.max_radius_depth_offset, 
                                                          stride_depth_offset=rm.stride_depth_offset, 
                                                          use_random_seed=rm.use_random_seed, 
                                                          image_size=rm.image_size)
            if self.cfg.use_depth:
                depth_model_path = self.tmp_dir / (bname + '_depth_model.bin')
                dm = self.cfg.depth_model
                self.depth_models[bname] = pyicg.DepthModel(bname + '_depth_model', body, depth_model_path.as_posix(),
                                                            sphere_radius=dm.sphere_radius, 
                                                            n_divides=dm.n_divides, 
                                                            n_points=dm.n_points, 
                                                            max_radius_depth_offset=dm.max_radius_depth_offset, 
                                                            stride_depth_offset=dm.stride_depth_offset, 
                                                            use_random_seed=dm.use_random_seed, 
                                                            image_size=dm.image_size)

            # Q: Possible to create on the fly?
            self.region_modalities[bname] = pyicg.RegionModality(bname + '_region_modality', body, self.color_camera, self.region_models[bname])
            
            self.region_modalities[bname].scales = self.cfg.region_scales
            self.region_modalities[bname].standard_deviations = self.cfg.region_standard_deviations

            if self.cfg.use_depth:
                self.depth_modalities[bname] = pyicg.DepthModality(bname + '_depth_modality', body, self.depth_camera, self.depth_models[bname])
                self.depth_modalities[bname].considered_distances = self.cfg.depth_considered_distances
                self.depth_modalities[bname].standard_deviations = self.cfg.depth_standard_deviations

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

    def set_all_bodies_behind_camera(self, except_object_names=None):
        T_bc_back = np.eye(4)
        T_bc_back[2,3] = -100
        for object_name, body in self.bodies.items():
            if except_object_names is None or object_name not in except_object_names:
                body.body2world_pose = T_bc_back

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

        return bodies, object_files

    def detected_bodies(self, detections: dict[str, np.array], scores=None):
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
        # print('detections:', detections)
        # print('scores:', scores)

        # Implementation 1: just update the current estimates, 1 object per cat

        # Init or scores are None -> update all without rules
        # if len(self.active_tracks) == 0 or scores is None:
        self.set_all_bodies_behind_camera()
        self.active_tracks = {}
        for obj_name, T_co in detections.items():
            if obj_name not in self.bodies:
                continue 
            
            score = scores[obj_name] if scores is not None else 1.0
            self.active_tracks[obj_name] = score
            self.bodies[obj_name].body2world_pose = T_co

        # # Implementation 2: more logic to reject bad detections if tracks are good enough
        # else:
        #     THRESH = 0.0
        #     MIN_SCORE_ACTIVE_TRACKS = 0.0

        #     # compare new detections score with active tracks score: 
        #     # if score lower by THRESH to previous det, do not take into account 
        #     object_to_remove_from_tracks_if_present = set(self.bodies.keys())

        #     for obj_name, T_co in detections.items():
        #         if obj_name not in self.bodies:
        #             continue 

        #         score = scores[obj_name] if scores is not None else 1.0
        #         if obj_name not in self.active_tracks:
        #             if score > MIN_SCORE_ACTIVE_TRACKS:
        #                 self.active_tracks[obj_name] = score
        #                 self.bodies[obj_name].body2world_pose = T_co
        #                 object_to_remove_from_tracks_if_present.remove(obj_name)
                
        #         elif score > self.active_tracks[obj_name] - THRESH:
        #             # score needs to be good enough with respect to 
        #             # previously recorded score for an update to happen
        #             self.active_tracks[obj_name] = score
        #             self.bodies[obj_name].body2world_pose = T_co
        #             object_to_remove_from_tracks_if_present.remove(obj_name)

        #     for obj_name in object_to_remove_from_tracks_if_present:
        #         if obj_name in self.active_tracks:
        #             del self.active_tracks[obj_name]


        
        
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

        t = time.perf_counter()
        self.tracker.ExecuteTrackingCycle(self.iteration)
        dt = time.perf_counter() - t

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
        t = time.perf_counter()
        self.tracker.UpdateViewers(self.iteration)
        dt = time.perf_counter() - t
        return dt

    def get_current_preds(self):
        preds = {}
        for obj_name in self.active_tracks.keys():
            preds[obj_name] = self.bodies[obj_name].body2world_pose

        return preds, self.active_tracks


