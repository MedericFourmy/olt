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
import pprint


from olt.config import TrackerConfig, CameraConfig, ModelConfig, RegionModalityConfig, DepthModalityConfig


class Tracker:
    """
    params: TODO
    """
    def __init__(self, 
                 obj_model_dir: Union[str,Path],
                 accepted_objs: Union[Set[str],str],
                 cfg: TrackerConfig
                 ) -> None:
        
        self.cfg = cfg
        self.tmp_dir = Path(self.cfg.tmp_dir_name)
        self.obj_model_dir = Path(obj_model_dir)
        self.accepted_objs = accepted_objs

        # detected bodies logic
        self.active_tracks = {}
        self.undetected_tracks = {}

        self.images_set = False

        # to send objects behind the camera
        # TODO: figure out other init
        self.T_bc_back = np.eye(4)
        self.T_bc_back[2,3] = -100

    def init(self):
        # print_cfg(self.cfg)
        assert_cfg_consistency(self.cfg)
    
        # Check if paths exist
        if not self.tmp_dir.exists(): self.tmp_dir.mkdir(parents=True)
        imgs_dir = self.tmp_dir / 'imgs'
        if imgs_dir.exists(): 
            shutil.rmtree(imgs_dir.as_posix(), ignore_errors=True)
        imgs_dir.mkdir(parents=True, exist_ok=True)

        assert(self.obj_model_dir.exists())

        # Main class
        self.tracker = pym3t.Tracker('tracker', synchronize_cameras=False)
        
        # Class dealing with OpenGL interface
        self.renderer_geometry = pym3t.RendererGeometry('renderer_geometry')

        # Cameras
        self.color_cameras = create_color_cameras(self.cfg.cameras)
        self.depth_cameras = create_depth_cameras(self.cfg.cameras)
        if len(self.depth_cameras) > 0:
            self.use_depth = True

        # Viewers
        if self.cfg.viewer_display or self.cfg.viewer_save:
            for color_cam in self.color_cameras.values():
                viewer = create_color_viewer(self.cfg, color_cam, self.renderer_geometry, imgs_dir)
                self.tracker.AddViewer(viewer)

            for depth_cam in self.depth_cameras.values():
                viewer = create_depth_viewer(self.cfg, depth_cam, self.renderer_geometry, imgs_dir)
                self.tracker.AddViewer(viewer)

        # bodies: create 1 for each object model with body names = body ids
        self.bodies, self.links, self.object_files = self.create_bodies(self.obj_model_dir, self.accepted_objs, self.cfg.geometry_unit_in_meter)
        # self.set_all_bodies_behind_camera()

        # Store collections of important objects
        self.region_models =  {}  # {body_name: RegionModel}
        self.depth_models =  {}   # {body_name: DepthModel}
        self.region_modalities = {}  # [{}] 
        self.depth_modalities = {} 
        self.optimizers = {}

        # At least 1 camera provides modality
        self.use_region_modality = len(self.cfg.region_modalities) > 0
        self.use_depth_modality = len(self.cfg.depth_modalities) > 0

        # Depth renderers to model occlusion
        self.occl_color_renderers = {}
        self.occl_depth_renderers = {}
        for i, rmc in self.cfg.region_modalities.items(): 
            if rmc.model_occlusions:
                color_cam = self.color_cameras[i]
                self.occl_color_renderers[i] = pym3t.FocusedBasicDepthRenderer(f'focused_color_depth_renderer_{color_cam.name}', self.renderer_geometry, color_cam)
        for i, dmc in self.cfg.depth_modalities.items(): 
            if dmc.model_occlusions:            
                depth_cam = self.depth_cameras[i]
                self.occl_depth_renderers[i] = pym3t.FocusedBasicDepthRenderer(f'focused_depth_depth_renderer_{depth_cam.name}', self.renderer_geometry, depth_cam)

        # Creating models and modalities for all specified objects
        for bname in self.bodies:
            body = self.bodies[bname]
            link = self.links[bname]

            self.renderer_geometry.AddBody(body)
            
            if self.use_region_modality:
                self.region_models[bname] = create_region_model(body, self.cfg.region_model, self.tmp_dir)
            if self.use_depth_modality:
                self.depth_models[bname] = create_depth_model(body, self.cfg.depth_model, self.tmp_dir)

            for i, rmc in self.cfg.region_modalities.items():
                depth_cam = self.depth_cameras[i] if i in self.depth_cameras else None 
                occl_renderer = self.self.occl_color_renderers[i] if i in self.occl_color_renderers else None 
                region_modality = create_region_modality(self.region_models[bname], body, self.color_cameras[i], rmc, depth_cam, occl_renderer)
                link.AddModality(region_modality)

            for i, dmc in self.cfg.depth_modalities.items():
                occl_renderer = self.self.occl_depth_renderers[i] if i in self.occl_depth_renderers else None 
                depth_modality = create_depth_modality(self.depth_models[bname], body, self.depth_cameras[i], dmc, occl_renderer)
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
        object_poses: list of object_name,pose pairs, pose expressed in world frame.
        {
            'obj_000010': pose1,
            'obj_000016': pose2,
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

    def set_images(self, color_imgs: Dict[int,np.array], depth_imgs: Dict[int,np.array] = {}):
        for i, color in color_imgs.items():
            if i not in self.color_cameras:
                print(f'Tracker.set_images: not color camera {i}')
            else:
                self.color_cameras[i].image = color
        for i, depth in depth_imgs.items():
            if i not in self.depth_cameras:
                print(f'Tracker.set_images: not depth camera {i}')
            else:
                self.depth_cameras[i].image = depth

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


def print_cfg(cfg):
    print('\n================\nInitializing tracker:')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)
    breakpoint()
    print('===========\n')


def assert_cfg_consistency(cfg: TrackerConfig):
    assert len(cfg.cameras) > 0, 'Need to provide at least one camera'
    for i, cam_cfg in cfg.cameras.items():
        assert (cam_cfg.color_intrinsics is None) == (cam_cfg.color2world_pose is None), f'unconsistent cam {i} color config'
        assert (cam_cfg.depth_intrinsics is None) == (cam_cfg.depth2world_pose is None), f'unconsistent cam {i} depth config'
    
    for i in cfg.region_modalities:
        assert i in cfg.cameras, f'camera {i} with no config'
        assert use_color(cfg.cameras[i]), f'RegionModality requires camera {i} with color config'
    for i in cfg.depth_modalities:
        assert i in cfg.cameras, f'camera {i} with no config'
        assert use_depth(cfg.cameras[i]), f'DepthModality requires camera {i} with color config'


def use_color(cam_cfg: CameraConfig):
    return cam_cfg.color_intrinsics is not None


def use_depth(cam_cfg: CameraConfig):
    return cam_cfg.depth_intrinsics is not None


def create_color_cameras(cameras_cfg: Dict[int,CameraConfig]):
    return {i: create_color_camera(cam_cfg, i) for i, cam_cfg in cameras_cfg.items() if use_color(cam_cfg)}


def create_depth_cameras(cameras_cfg: Dict[int,CameraConfig]):
    return {i: create_depth_camera(cam_cfg, i) for i, cam_cfg in cameras_cfg.items() if use_depth(cam_cfg)}


def create_color_camera(cam_cfg: CameraConfig, idx):
    color_camera = pym3t.DummyColorCamera(f'cam_color_{idx}')
    color_camera.camera2world_pose = cam_cfg.color2world_pose
    color_camera.intrinsics = pym3t.Intrinsics(**cam_cfg.color_intrinsics)
    return color_camera


def create_depth_camera(cam_cfg: CameraConfig, idx):
    depth_camera = pym3t.DummyDepthCamera(f'cam_depth_{idx}')
    depth_camera.camera2world_pose = cam_cfg.depth2world_pose
    depth_camera.intrinsics = pym3t.Intrinsics(**cam_cfg.depth_intrinsics)
    return depth_camera


def create_color_viewer(cfg: TrackerConfig, color_camera: pym3t.DummyColorCamera, 
                 renderer_geometry: pym3t.RendererGeometry, imgs_dir: Path):
    viewer = pym3t.NormalColorViewer(f'normal_viewer_{color_camera.name}', color_camera, renderer_geometry)
    if cfg.viewer_save: viewer.StartSavingImages(imgs_dir.as_posix(), 'png')
    viewer.set_opacity(0.5)  # [0.0-1.0]
    viewer.display_images = cfg.viewer_display
    return viewer


def create_depth_viewer(cfg: TrackerConfig, depth_camera: pym3t.DummyDepthCamera, 
                 renderer_geometry: pym3t.RendererGeometry, imgs_dir: Path):
    viewer = pym3t.NormalColorViewer(f'normal_viewer_{depth_camera.name}', depth_camera, renderer_geometry)
    if cfg.viewer_save: viewer.StartSavingImages(imgs_dir.as_posix(), 'png')
    viewer.set_opacity(0.5)  # [0.0-1.0]
    viewer.display_images = cfg.viewer_display
    return viewer


def create_region_model(body: pym3t.Body, model_cfg: ModelConfig, tmp_dir: Path):
    region_model_path = tmp_dir / (body.name + '_region_model.bin')
    mc = model_cfg
    return pym3t.RegionModel(body.name + '_region_model', body, region_model_path.as_posix(),
                                sphere_radius=mc.sphere_radius, 
                                n_divides=mc.n_divides, 
                                n_points_max=mc.n_points_max, 
                                max_radius_depth_offset=mc.max_radius_depth_offset, 
                                stride_depth_offset=mc.stride_depth_offset, 
                                use_random_seed=mc.use_random_seed, 
                                image_size=mc.image_size)


def create_depth_model(body: pym3t.Body, model_cfg: ModelConfig, tmp_dir: Path):
    region_model_path = tmp_dir / (body.name + '_region_model.bin')
    mc = model_cfg
    return pym3t.DepthModel(body.name + '_depth_model', body, region_model_path.as_posix(),
                            sphere_radius=mc.sphere_radius, 
                            n_divides=mc.n_divides, 
                            n_points_max=mc.n_points_max, 
                            max_radius_depth_offset=mc.max_radius_depth_offset, 
                            stride_depth_offset=mc.stride_depth_offset, 
                            use_random_seed=mc.use_random_seed, 
                            image_size=mc.image_size)


def create_region_modality(region_model: pym3t.RegionModel, body: pym3t.Body, 
                           color_camera: pym3t.DummyColorCamera, cfg_rm: RegionModalityConfig, 
                           depth_camera: pym3t.DummyDepthCamera=None, 
                           focused_color_depth_renderer: pym3t.FocusedBasicDepthRenderer=None):

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

    if cfg_rm.model_occlusions and focused_color_depth_renderer is not None:  
        region_modality.ModelOcclusions(focused_color_depth_renderer)
        region_modality.modeled_depth_offset_radius = cfg_rm.modeled_depth_offset_radius
        region_modality.modeled_occlusion_radius = cfg_rm.modeled_occlusion_radius
        region_modality.modeled_occlusion_threshold = cfg_rm.modeled_occlusion_threshold

    return region_modality


def create_depth_modality(depth_model: pym3t.DepthModel, body: pym3t.Body, 
                          depth_camera: pym3t.DummyDepthCamera, cfg_dm: DepthModalityConfig,
                          focused_depth_depth_renderer: pym3t.FocusedBasicDepthRenderer=None):
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

    if cfg_dm.model_occlusions and focused_depth_depth_renderer is not None:  
        depth_modality.ModelOcclusions(focused_depth_depth_renderer)
        depth_modality.modeled_depth_offset_radius = cfg_dm.modeled_depth_offset_radius
        depth_modality.modeled_occlusion_radius = cfg_dm.modeled_occlusion_radius
        depth_modality.modeled_occlusion_threshold = cfg_dm.modeled_occlusion_threshold

    return depth_modality