"""

Vocabulary
- body name: name of a specific instance of a object in pym3t

Notations
- T_co: SE(3) transformation that translate and rotate a vector from frame o to frame c: c_v = T_co @ o_v
"""


import time
import numpy as np
from pathlib import Path
import shutil
from typing import Union, Set, Dict, List
import pym3t


from olt.config import TrackerConfig, CameraConfig, RegionModalityConfig, DepthModalityConfig


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
                 accepted_objs: Union[Set[str],str],
                 cfg: TrackerConfig,
                 cameras_cfg: List[CameraConfig],
                 ) -> None:
        
        self.cfg = cfg
        self.cameras_cfg = cameras_cfg
        self.tmp_dir = Path(self.cfg.tmp_dir_name)
        self.obj_model_dir = Path(obj_model_dir)
        self.accepted_objs = accepted_objs

        # some other parameters
        self.geometry_unit_in_meter_ycbv_urdf = 0.001

        # detected bodies logic
        self.active_tracks = {}
        self.undetected_tracks = {}

        self.images_set = False

        # to send objects behind the camera
        self.T_bc_back = np.eye(4)
        self.T_bc_back[2,3] = -100

    def init(self):
        # Check if paths exist
        if not self.tmp_dir.exists(): self.tmp_dir.mkdir(parents=True)
        imgs_dir = self.tmp_dir / 'imgs'
        # erase image directory
        if imgs_dir.exists(): 
            shutil.rmtree(imgs_dir.as_posix(), ignore_errors=True)
        imgs_dir.mkdir(parents=True, exist_ok=True)

        assert(self.obj_model_dir.exists())

        # Main class
        self.tracker = pym3t.Tracker('tracker', synchronize_cameras=False)
        
        # Renderer for preprocessing
        self.renderer_geometry = pym3t.RendererGeometry('renderer_geometry')

        # Cameras
        self.color_cameras = [create_color_camera(cam_cfg, i) for 
                              i, cam_cfg in enumerate(self.cameras_cfg)]
        if self.cfg.use_depth:
            self.color_cameras = [create_depth_camera(cam_cfg, i) for 
                                  i, cam_cfg in enumerate(self.cameras_cfg)]

        # Viewers
        if self.cfg.viewer_display or self.cfg.viewer_save:
            self.color_viewers = [create_color_viewer(self.cfg, color_cam, self.renderer_geometry, imgs_dir)
                                  for color_cam in self.color_cameras]
            for viewer in self.color_viewers:
                self.tracker.AddViewer(viewer)
            
            if self.cfg.use_depth and self.cfg.display_depth:
                self.depth_viewers = [create_depth_viewer(self.cfg, depth_cam, self.renderer_geometry, imgs_dir)
                                      for depth_cam in self.depth_cameras]
                for viewer in self.depth_viewers:
                    self.tracker.AddViewer(viewer)

        # bodies: create 1 for each object model with body names = body ids
        self.bodies, self.links, self.object_files = self.create_bodies(self.obj_model_dir, self.accepted_objs, self.geometry_unit_in_meter_ycbv_urdf)
        self.set_all_bodies_behind_camera()

        self.region_models = {}
        self.depth_models = {}
        self.region_modalities_lst = {} 
        self.depth_modalities_lst = {} 
        self.optimizers = {}

        for bname in self.bodies:
            body = self.bodies[bname]
            link = self.links[bname]

            self.renderer_geometry.AddBody(body)

            region_model_path = self.tmp_dir / (bname + '_region_model.bin')
            rm = self.cfg.region_model
            self.region_models[bname] = pym3t.RegionModel(bname + '_region_model', body, region_model_path.as_posix(),
                                                          sphere_radius=rm.sphere_radius, 
                                                          n_divides=rm.n_divides, 
                                                          n_points_max=rm.n_points_max, 
                                                          max_radius_depth_offset=rm.max_radius_depth_offset, 
                                                          stride_depth_offset=rm.stride_depth_offset, 
                                                          use_random_seed=rm.use_random_seed, 
                                                          image_size=rm.image_size)

            if self.cfg.use_depth and not self.cfg.no_depth_modality:
                depth_model_path = self.tmp_dir / (bname + '_depth_model.bin')
                dm = self.cfg.depth_model
                self.depth_models[bname] = pym3t.DepthModel(bname + '_depth_model', body, depth_model_path.as_posix(),
                                                            sphere_radius=dm.sphere_radius, 
                                                            n_divides=dm.n_divides, 
                                                            n_points_max=dm.n_points_max, 
                                                            max_radius_depth_offset=dm.max_radius_depth_offset, 
                                                            stride_depth_offset=dm.stride_depth_offset, 
                                                            use_random_seed=dm.use_random_seed, 
                                                            image_size=dm.image_size)



            self.region_modalities_lst[bname] = [create_region_modality(self.region_models[bname], body, color_cam, self.cfg.region_modality)
                                                    for color_cam in self.color_cameras]

            if self.cfg.use_depth and not self.cfg.no_depth_modality:
                self.region_modalities_lst[bname] = [create_depth_modality(self.depth_models[bname], body, depth_cam, self.cfg.depth_modality)
                                                         for depth_cam in self.depth_cameras]
            
            for region_modality in self.region_modalities_lst[bname]:
                link.AddModality(region_modality)
                if self.cfg.use_depth and not self.cfg.no_depth_modality:
                    for depth_modality in self.depth_modalities_lst[bname]:
                        link.AddModality(depth_modality)

            # Add remove optimizers?
            self.optimizers[bname] = pym3t.Optimizer(bname+'_optimizer', link, self.cfg.tikhonov_parameter_rotation, self.cfg.tikhonov_parameter_translation)
            self.tracker.AddOptimizer(self.optimizers[bname])

        # execute tracking loop
        self.tracker.n_corr_iterations = self.cfg.n_corr_iterations
        self.tracker.n_update_iterations = self.cfg.n_update_iterations

        print('SETUP TRACKER')
        ok = self.tracker.SetUp()
        if not ok:
            raise(ValueError('Error in SetUp'))
        print('TRACKER SETUP OK')

        self.iteration = 0
    
    def set_body_behind_camera(self, oname):
        assert oname in self.bodies
        self.bodies[oname].body2world_pose = self.T_bc_back
        self.links[oname].link2world_pose = self.T_bc_back
    
    def set_all_bodies_behind_camera(self):
        for oname in self.bodies:
            self.set_body_behind_camera(oname)

    def create_bodies(self, 
                    object_model_dir: Path, 
                    accepted_objs: Union[Set[str],str], 
                    geometry_unit_in_meter: float) -> Dict[str, pym3t.Body]:
        
        # Bodies
        object_files = {}
        for obj_dir in object_model_dir.iterdir():
            obj_files = list(obj_dir.glob('*.obj'))
            if len(obj_files) == 1:
                obj_path = obj_files[0]
                object_files[obj_dir.name] = obj_path
            else:
                print('PROBLEM: less or more than one file were found')
        
        # obj_name: 'obj_'
        bodies = {
            obj_name: pym3t.Body(
                name=obj_name,
                geometry_path=obj_path.as_posix(),
                geometry_unit_in_meter=geometry_unit_in_meter,
                geometry_counterclockwise=True,
                geometry_enable_culling=True,
                geometry2body_pose=np.eye(4)
            )
            for obj_name, obj_path in object_files.items()
            if accepted_objs == 'all' or obj_name in accepted_objs
        }

        links = {
            body_name: pym3t.Link(body_name + '_link', body)
            for body_name, body in bodies.items()
        }

        return bodies, links, object_files

    def detected_bodies(self, object_poses: Dict[str, np.array], scores=None, reset_after_n=0):
        """
        object_poses: list of object name,pose pairs coming from a pose estimator like happy pose
        Multiple objects of the sane
        {
            'obj_000010': pose1,
            'obj_000016': pose2,
            'obj_000010': pose3,
            ...
        }

        scores: object_name, score dict
        reset_after_n: number of time an active track needs to NOT be detected to be removed from active tracks

        FOR NOW: assume unique objects and reinitialize there pose
        """

        # Pre-filter detections with accepted objects
        if self.accepted_objs != 'all':
            object_poses = {oname: object_poses[oname] for oname in self.accepted_objs if oname in object_poses}
            if scores is not None:
                scores = {oname: scores[oname] for oname in self.accepted_objs if oname in scores}

        # Init or scores are None -> update all without rules
        # if len(self.active_tracks) == 0 or scores is None:
        if reset_after_n == 0:
            self.set_all_bodies_behind_camera()
            self.active_tracks = {}
        
            for obj_name, T_co in object_poses.items():
                if obj_name not in self.bodies:
                    continue 
                
                score = scores[obj_name] if scores is not None else 1.0
                self.active_tracks[obj_name] = score
                self.bodies[obj_name].body2world_pose = T_co
                self.links[obj_name].link2world_pose = T_co

        else:
            # check if some tracks are lost and decide to keep them active or not
            tracks_to_remove = []
            for obj_name in self.active_tracks:
                if obj_name not in object_poses:
                    # Rule: if object name not detected for a while, consider lost
                    if obj_name not in self.undetected_tracks:
                        self.undetected_tracks[obj_name] = 1
                    else:
                        self.undetected_tracks[obj_name] += 1
                    if self.undetected_tracks[obj_name] >= reset_after_n:
                        self.set_body_behind_camera(obj_name)
                        tracks_to_remove.append(obj_name)
            # print('self.undetected_tracks\n', self.undetected_tracks)

            for obj_name in tracks_to_remove:
                # print('Remove track: ', obj_name)
                del self.active_tracks[obj_name]
                del self.undetected_tracks[obj_name]

            # Create/Update active tracks
            for obj_name, T_co in object_poses.items():
                if obj_name not in self.bodies:
                    continue 

                score = scores[obj_name] if scores is not None else 1.0
                self.active_tracks[obj_name] = score
                self.bodies[obj_name].body2world_pose = T_co
                self.links[obj_name].link2world_pose = T_co
                if obj_name in self.undetected_tracks:
                    del self.undetected_tracks[obj_name]

    def set_images(self, color_lst: np.array, depth_lst: Union[List[np.array],None] = None):
        if depth_lst is not None:
            assert len(color_lst) == len(depth_lst)
        for i in range(len(color_lst)):
            self.color_cameras[i].image = color_lst[i]
            if self.cfg.use_depth:
                self.depth_cameras[i].image = depth_lst[i]

        # verifying the images have been properly setup
        ok = self.tracker.UpdateCameras(True) 
        if not ok:
            raise ValueError('Something is wrong with the provided images')
        self.images_set = True

    def track(self):
        assert self.images_set, "No image was set yet! Call tracker.set_image(img)"
        if self.iteration == 0:
            # print('StartModalities!')
            # self.tracker.StartModalities(self.iteration)
            self.tracker.ExecuteStartingStep(self.iteration)
            

        t = time.perf_counter()
        self.tracker.ExecuteTrackingStep(self.iteration)
        dt = time.perf_counter() - t

        self.iteration += 1

        return dt

    def update_viewers(self, it=None):
        t = time.perf_counter()
        if it is None:
            it = self.iteration
        self.tracker.UpdateViewers(it)
        dt = time.perf_counter() - t
        return dt

    def get_current_preds(self):
        preds = {}
        for obj_name in self.active_tracks.keys():
            preds[obj_name] = self.bodies[obj_name].body2world_pose

        return preds, self.active_tracks





def create_color_camera(cam_cfg: CameraConfig, idx):
    color_camera = pym3t.DummyColorCamera(f'cam_color_{idx}')
    color_camera.camera2world_pose = cam_cfg.color2world_pose
    color_camera.intrinsics = pym3t.Intrinsics(**cam_cfg.rgb_intrinsics)
    return color_camera

def create_depth_camera(cam_cfg: CameraConfig, idx):
    assert cam_cfg.depth2world_pose is not None and cam_cfg.depth_intrinsics is not None, f"depth extrinsics or intrinsics not set {idx}"
    depth_camera = pym3t.DummyDepthCamera(f'cam_depth_{idx}')
    depth_camera.camera2world_pose = cam_cfg.depth2world_pose
    depth_camera.intrinsics = pym3t.Intrinsics(**cam_cfg.depth_intrinsics)
    return depth_camera

def create_color_viewer(cfg: TrackerConfig, color_camera: pym3t.DummyColorCamera, 
                 renderer_geometry: pym3t.RendererGeometry, imgs_dir: Path):
    viewer = pym3t.NormalColorViewer(f'normal_viewer_{color_camera.name}', color_camera, renderer_geometry)
    if cfg.viewer_save:
        viewer.StartSavingImages(imgs_dir.as_posix(), 'png')
    viewer.set_opacity(0.5)  # [0.0-1.0]
    viewer.display_images = cfg.viewer_display
    return viewer

def create_depth_viewer(cfg: TrackerConfig, depth_camera: pym3t.DummyDepthCamera, 
                 renderer_geometry: pym3t.RendererGeometry, imgs_dir: Path):
    viewer = pym3t.NormalColorViewer(f'normal_viewer_{depth_camera.name}', depth_camera, renderer_geometry)
    if cfg.viewer_save:
        viewer.StartSavingImages(imgs_dir.as_posix(), 'png')
    viewer.set_opacity(0.5)  # [0.0-1.0]
    viewer.display_images = cfg.viewer_display
    return viewer


def create_region_modality(region_model: pym3t.RegionModel, body: pym3t.Body, 
                           color_camera: pym3t.DummyColorCamera, cfg_rm: RegionModalityConfig, depth_camera=None):

    # Q: Possible to create on the fly?
    region_modality = pym3t.RegionModality(f'region_modality_{color_camera.name}_{body.name}', body, color_camera, region_model)

    # Parameters for general distribution
    region_modality.n_lines_max = cfg_rm.n_lines_max
    region_modality.use_adaptive_coverage = cfg_rm.use_adaptive_coverage
    region_modality.reference_contour_length = cfg_rm.reference_contour_length 
    region_modality.min_continuous_distance = cfg_rm.min_continuous_distance 
    region_modality.function_length = cfg_rm.function_length 
    region_modality.distribution_length = cfg_rm.distribution_length 
    region_modality.function_amplitude = cfg_rm.function_amplitude 
    region_modality.function_slope = cfg_rm.function_slope 
    region_modality.learning_rate = cfg_rm.learning_rate 
    region_modality.n_global_iterations = cfg_rm.n_global_iterations 
    region_modality.scales = cfg_rm.scales 
    region_modality.standard_deviations = cfg_rm.standard_deviations 
    
    # Parameters for histogram calculation
    region_modality.n_histogram_bins = cfg_rm.n_histogram_bins
    region_modality.learning_rate_f = cfg_rm.learning_rate_f
    region_modality.learning_rate_b = cfg_rm.learning_rate_b
    region_modality.unconsidered_line_length = cfg_rm.unconsidered_line_length
    region_modality.max_considered_line_length = cfg_rm.max_considered_line_length

    region_modality.visualize_pose_result = False
    region_modality.visualize_lines_correspondence = False
    region_modality.visualize_points_correspondence = False
    region_modality.visualize_points_depth_image_correspondence = False
    region_modality.visualize_points_depth_rendering_correspondence = False
    region_modality.visualize_points_result = False
    region_modality.visualize_points_histogram_image_result = False
    region_modality.visualize_points_histogram_image_optimization = False
    region_modality.visualize_points_optimization = False
    region_modality.visualize_gradient_optimization = False
    region_modality.visualize_hessian_optimization = False
    
    # Occlusion handling
    if cfg_rm.model_occlusions or (cfg_rm.measure_occlusions and depth_camera is not None):
        region_modality.min_n_unoccluded_points = cfg_rm.min_n_unoccluded_points
        region_modality.n_unoccluded_iterations = cfg_rm.n_unoccluded_iterations
    
    if cfg_rm.measure_occlusions and depth_camera is not None:
        region_modality.MeasureOcclusions(depth_camera)
        region_modality.measured_depth_offset_radius = cfg_rm.measured_depth_offset_radius
        region_modality.measured_occlusion_radius = cfg_rm.measured_occlusion_radius
        region_modality.measured_occlusion_threshold = cfg_rm.measured_occlusion_threshold

    # if cfg_rm.model_occlusions:  
    #     region_modality.ModelOcclusions(depth_renderer)
    #     region_modality.modeled_depth_offset_radius = cfg_rm.modeled_depth_offset_radius
    #     region_modality.modeled_occlusion_radius = cfg_rm.modeled_occlusion_radius
    #     region_modality.modeled_occlusion_threshold = cfg_rm.modeled_occlusion_threshold

    return region_modality


def create_depth_modality(depth_model: pym3t.DepthModel, body: pym3t.Body, 
                          depth_camera: pym3t.DummyDepthCamera, cfg_dm: DepthModalityConfig):
    depth_modality = pym3t.DepthModality(f'depth_modality_{depth_camera.name}_{body.name}', body, depth_camera, depth_model)

    # Parameters for general distribution
    depth_modality.n_points_max = cfg_dm.n_points_max
    depth_modality.use_adaptive_coverage = cfg_dm.use_adaptive_coverage
    depth_modality.reference_surface_area = cfg_dm.reference_surface_area
    depth_modality.stride_length = cfg_dm.stride_length
    depth_modality.considered_distances = cfg_dm.considered_distances
    depth_modality.standard_deviations = cfg_dm.standard_deviations

    depth_modality.visualize_correspondences_correspondence = False
    depth_modality.visualize_points_correspondence = False
    depth_modality.visualize_points_depth_rendering_correspondence = False
    depth_modality.visualize_points_optimization = False
    depth_modality.visualize_points_result = False

    # Occlusion handling
    if cfg_dm.model_occlusions or cfg_dm.measure_occlusions:
        depth_modality.min_n_unoccluded_points = cfg_dm.min_n_unoccluded_points
        depth_modality.n_unoccluded_iterations = cfg_dm.n_unoccluded_iterations
    
    if cfg_dm.measure_occlusions:
        depth_modality.MeasureOcclusions()
        depth_modality.measured_depth_offset_radius = cfg_dm.measured_depth_offset_radius
        depth_modality.measured_occlusion_radius = cfg_dm.measured_occlusion_radius
        depth_modality.measured_occlusion_threshold = cfg_dm.measured_occlusion_threshold

    # if cfg_dm.model_occlusions:  
    #     depth_modality.modeled_depth_offset_radius = cfg_dm.modeled_depth_offset_radius
    #     depth_modality.modeled_occlusion_radius = cfg_dm.modeled_occlusion_radius
    #     depth_modality.modeled_occlusion_threshold = cfg_dm.modeled_occlusion_threshold

    return depth_modality