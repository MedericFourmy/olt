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
from typing import Union, Set, Dict
import pym3t


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
                 accepted_objs: Union[Set[str],str],
                 cfg: TrackerConfig,
                 rgb_intrinsics: dict,
                 depth_intrinsics: Union[dict,None] = None,
                 color2depth_pose: Union[np.ndarray,None] = None,
                 ) -> None:
        
        if cfg.use_depth:
            assert depth_intrinsics is not None

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

        # detected bodies logic
        self.active_tracks = {}
        self.undetected_tracks = {}

        self.image_set = False

        # to send objects behind the camera
        self.T_bc_back = np.eye(4)
        self.T_bc_back[2,3] = -100

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
        self.tracker = pym3t.Tracker('tracker', synchronize_cameras=False)
        
        # Renderer for preprocessing
        self.renderer_geometry = pym3t.RendererGeometry('renderer geometry')

        self.color_camera = pym3t.DummyColorCamera('cam_color')
        self.color_camera.color2depth_pose = self.color2depth_pose
        self.color_camera.camera2world_pose = np.eye(4)  # color camera fixed at the origin of the world
        self.color_camera.intrinsics = pym3t.Intrinsics(**self.rgb_intrinsics)

        if self.cfg.use_depth:
            self.depth_camera = pym3t.DummyDepthCamera('cam_depth', depth_scale=self.cfg.depth_scale)
            self.depth_camera.color2depth_pose = self.color2depth_pose
            self.depth_camera.camera2world_pose = self.color_camera.depth2color_pose  # world shifted by depth2color transformation
            self.depth_camera.intrinsics = pym3t.Intrinsics(**self.depth_intrinsics)

        if self.cfg.viewer_display or self.cfg.viewer_save:
            #########################################
            # Viewers
            self.color_viewer = pym3t.NormalColorViewer('color_'+self.cfg.viewer_name, self.color_camera, self.renderer_geometry)
            if self.cfg.viewer_save:
                self.color_viewer.StartSavingImages(self.imgs_dir.as_posix(), 'png')
            self.color_viewer.set_opacity(0.5)  # [0.0-1.0]
            self.color_viewer.display_images = self.cfg.viewer_display
            self.tracker.AddViewer(self.color_viewer)

            if self.cfg.use_depth and self.cfg.display_depth:
                depth_viewer = pym3t.NormalDepthViewer('depth_'+self.cfg.viewer_name, self.depth_camera, self.renderer_geometry)
                if self.cfg.viewer_save:
                    depth_viewer.StartSavingImages(self.imgs_dir.as_posix(), 'png')
                depth_viewer.display_images = self.cfg.viewer_display
                self.tracker.AddViewer(depth_viewer)
            #########################################

        # bodies: create 1 for each object model with body names = body ids
        self.bodies, self.links, self.object_files = self.create_bodies(self.obj_model_dir, self.accepted_objs, self.geometry_unit_in_meter_ycbv_urdf)
        self.set_all_bodies_behind_camera()

        self.region_models = {}
        self.region_modalities = {} 
        self.depth_models = {}
        self.depth_modalities = {} 
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

            # Q: Possible to create on the fly?
            self.region_modalities[bname] = pym3t.RegionModality(bname + '_region_modality', body, self.color_camera, self.region_models[bname])

            # Parameters for general distribution
            self.region_modalities[bname].n_lines_max = self.cfg.region_modality.n_lines_max
            self.region_modalities[bname].use_adaptive_coverage = self.cfg.region_modality.use_adaptive_coverage
            self.region_modalities[bname].reference_contour_length = self.cfg.region_modality.reference_contour_length 
            self.region_modalities[bname].min_continuous_distance = self.cfg.region_modality.min_continuous_distance 
            self.region_modalities[bname].function_length = self.cfg.region_modality.function_length 
            self.region_modalities[bname].distribution_length = self.cfg.region_modality.distribution_length 
            self.region_modalities[bname].function_amplitude = self.cfg.region_modality.function_amplitude 
            self.region_modalities[bname].function_slope = self.cfg.region_modality.function_slope 
            self.region_modalities[bname].learning_rate = self.cfg.region_modality.learning_rate 
            self.region_modalities[bname].n_global_iterations = self.cfg.region_modality.n_global_iterations 
            self.region_modalities[bname].scales = self.cfg.region_modality.scales 
            self.region_modalities[bname].standard_deviations = self.cfg.region_modality.standard_deviations 
            
            # Parameters for histogram calculation
            self.region_modalities[bname].n_histogram_bins = self.cfg.region_modality.n_histogram_bins
            self.region_modalities[bname].learning_rate_f = self.cfg.region_modality.learning_rate_f
            self.region_modalities[bname].learning_rate_b = self.cfg.region_modality.learning_rate_b
            self.region_modalities[bname].unconsidered_line_length = self.cfg.region_modality.unconsidered_line_length
            self.region_modalities[bname].max_considered_line_length = self.cfg.region_modality.max_considered_line_length

            self.region_modalities[bname].visualize_pose_result = False
            self.region_modalities[bname].visualize_lines_correspondence = False
            self.region_modalities[bname].visualize_points_correspondence = False
            self.region_modalities[bname].visualize_points_depth_image_correspondence = False
            self.region_modalities[bname].visualize_points_depth_rendering_correspondence = False
            self.region_modalities[bname].visualize_points_result = False
            self.region_modalities[bname].visualize_points_histogram_image_result = False
            self.region_modalities[bname].visualize_points_histogram_image_optimization = False
            self.region_modalities[bname].visualize_points_optimization = False
            self.region_modalities[bname].visualize_gradient_optimization = False
            self.region_modalities[bname].visualize_hessian_optimization = False
            

            if self.cfg.use_depth:

                if self.cfg.measure_occlusions:# or self.cfg.model_occlusions
                    self.region_modalities[bname].min_n_unoccluded_lines = self.cfg.region_modality.min_n_unoccluded_lines
                    self.region_modalities[bname].n_unoccluded_iterations = self.cfg.region_modality.n_unoccluded_iterations
                    self.region_modalities[bname].MeasureOcclusions(self.depth_camera)
                    # Parameters for occlusion handling
                    self.region_modalities[bname].measured_depth_offset_radius = self.cfg.region_modality.measured_depth_offset_radius
                    self.region_modalities[bname].measured_occlusion_radius = self.cfg.region_modality.measured_occlusion_radius
                    self.region_modalities[bname].measured_occlusion_threshold = self.cfg.region_modality.measured_occlusion_threshold

                if not self.cfg.no_depth_modality:
                    self.depth_modalities[bname] = pym3t.DepthModality(bname + '_depth_modality', body, self.depth_camera, self.depth_models[bname])

                    # Parameters for general distribution
                    self.depth_modalities[bname].n_points_max = self.cfg.depth_modality.n_points_max
                    self.depth_modalities[bname].use_adaptive_coverage = self.cfg.depth_modality.use_adaptive_coverage
                    self.depth_modalities[bname].reference_surface_area = self.cfg.depth_modality.reference_surface_area
                    self.depth_modalities[bname].stride_length = self.cfg.depth_modality.stride_length
                    self.depth_modalities[bname].considered_distances = self.cfg.depth_modality.considered_distances
                    self.depth_modalities[bname].standard_deviations = self.cfg.depth_modality.standard_deviations

                    self.depth_modalities[bname].visualize_correspondences_correspondence = False
                    self.depth_modalities[bname].visualize_points_correspondence = False
                    self.depth_modalities[bname].visualize_points_depth_rendering_correspondence = False
                    self.depth_modalities[bname].visualize_points_optimization = False
                    self.depth_modalities[bname].visualize_points_result = False

                    if self.cfg.measure_occlusions:# or self.cfg.model_occlusions
                        self.depth_modalities[bname].min_n_unoccluded_points = self.cfg.depth_modality.min_n_unoccluded_points
                        self.depth_modalities[bname].n_unoccluded_iterations = self.cfg.depth_modality.n_unoccluded_iterations
                        self.depth_modalities[bname].measured_occlusion_radius = self.cfg.depth_modality.measured_occlusion_radius
                        self.depth_modalities[bname].measured_occlusion_threshold = self.cfg.depth_modality.measured_occlusion_threshold


                # TODO: Model occlusions
                # if self.cfg.model_occlusions:  
                #     self.region_modalities[bname].modeled_depth_offset_radius = self.cfg.region_modality.modeled_depth_offset_radius
                #     self.region_modalities[bname].modeled_occlusion_radius = self.cfg.region_modality.modeled_occlusion_radius
                #     self.region_modalities[bname].modeled_occlusion_threshold = self.cfg.region_modality.modeled_occlusion_threshold

            link.AddModality(self.region_modalities[bname])
            if self.cfg.use_depth and not self.cfg.no_depth_modality:
                link.AddModality(self.region_modalities[bname])

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
            # print('StartModalities!')
            # self.tracker.StartModalities(self.iteration)
            print('ExecuteStartingStep!')
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


