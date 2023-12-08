import os
from pathlib import Path
from dataclasses import dataclass, field
import logging

DATASET_NAMES = ['ycbv', 'rotd']

HAPPYPOSE_DATA_DIR = os.environ.get('HAPPYPOSE_DATA_DIR')
assert len(HAPPYPOSE_DATA_DIR) > 0, 'HAPPYPOSE_DATA_DIR env variable is not set'
HAPPYPOSE_DATA_DIR = Path(HAPPYPOSE_DATA_DIR)

BOP_DS_DIRS = {ds_name: HAPPYPOSE_DATA_DIR / Path('bop_datasets') / ds_name for ds_name in DATASET_NAMES}
OBJ_MODEL_DIRS = {ds_name: HAPPYPOSE_DATA_DIR / Path('urdfs') / ds_name for ds_name in DATASET_NAMES}

# Adapted from https://stackoverflow.com/questions/53632152/why-cant-dataclasses-have-mutable-defaults-in-their-class-attributes-declaratio
default_list = lambda l: field(default_factory=lambda: l)



@dataclass
class ModelConfig:
    sphere_radius: float = 0.8
    n_divides: int = 4 
    n_points_max: int = 200
    max_radius_depth_offset: float = 0.05
    stride_depth_offset: float = 0.002 
    use_random_seed: bool = False
    image_size: int = 2000


@dataclass
class RegionModalityConfig:

    # Parameters for general distribution
    n_lines_max: int = 200
    use_adaptive_coverage: bool = False
    reference_contour_length: float = 0.0
    min_continuous_distance: float = 3.0
    function_length: int = 8
    distribution_length: int = 12
    function_amplitude: float = 0.43
    function_slope: float = 0.5
    learning_rate: float = 1.3
    n_global_iterations: int = 1
    scales: list[float] = default_list([6, 4, 2, 1]) 
    standard_deviations: list[float] = default_list([15.0, 5.0, 3.5, 1.5])

    # Parameters for histogram calculation
    n_histogram_bins: int = 16
    learning_rate_f: float = 0.2
    learning_rate_b: float = 0.2
    unconsidered_line_length: float = 0.5
    max_considered_line_length: float = 20.0

    # Parameters for occlusion handling
    measure_occlusions: bool = False
    measured_depth_offset_radius: float = 0.01
    measured_occlusion_radius: float = 0.01
    measured_occlusion_threshold: float = 0.03
    model_occlusions: bool = False
    modeled_depth_offset_radius: float = 0.01
    modeled_occlusion_radius: float = 0.01
    modeled_occlusion_threshold: float = 0.03
    n_unoccluded_iterations: int = 10
    min_n_unoccluded_lines: int = 0


@dataclass
class DepthModalityConfig:
    # Parameters for general distribution
    n_points_max: int = 200
    use_adaptive_coverage: bool = False
    reference_surface_area: float = 0.0
    stride_length: float = 0.005
    considered_distances: list[float] = default_list([0.05, 0.02, 0.01])
    standard_deviations: list[float] = default_list([0.05, 0.03, 0.02])

    # Parameters for occlusion handling
    measure_occlusions: bool = False
    measured_depth_offset_radius: float = 0.01
    measured_occlusion_radius: float = 0.01
    measured_occlusion_threshold: float = 0.03
    model_occlusions: bool = False
    modeled_depth_offset_radius: float = 0.01
    modeled_occlusion_radius: float = 0.01
    modeled_occlusion_threshold: float = 0.03
    n_unoccluded_iterations: int = 10
    min_n_unoccluded_points: int = 0


"""
Default value taken from M3T YCBV evaluation
"""
@dataclass
class TrackerConfig:
    tmp_dir_name: str = 'tmp'
    viewer_name: str = 'normal_viewer'
    use_depth: bool = False
    viewer_display: bool = False
    display_depth: bool = False
    viewer_save: bool = False
    no_depth_modality: bool = False
    depth_scale: float = 0.001
    measure_occlusions: bool = False
    
    # Optimization params
    n_corr_iterations: int = 4
    n_update_iterations: int = 2
    tikhonov_parameter_rotation: float = 1000.0
    tikhonov_parameter_translation: float = 30000.0

    # Models params
    region_model: ModelConfig = ModelConfig()
    depth_model: ModelConfig = ModelConfig()

    # Modalities params
    region_modality: RegionModalityConfig = RegionModalityConfig()
    depth_modality: DepthModalityConfig = DepthModalityConfig()

@dataclass
class LocalizerConfig:
    detector_threshold: float = 0.8
    n_workers: int = 1  # 0 not possible with 'panda'
    renderer_name: str = 'bullet'  # options: 'bullet', 'panda'
    n_coarse: int = 1 
    n_refiner: int = 3
    training_type: str = 'pbr'  # pbr or synt+real

@dataclass
class EvaluationBOPConfig:
    tracker_cfg: TrackerConfig = TrackerConfig()
    localizer_cfg: LocalizerConfig = LocalizerConfig()
    ds_name: str = 'ycbv'


logcfg = { 'version': 1,
           'formatters': {
               'normal': {
                   'format': '%(levelname)-8s %(message)s'}},
           'handlers': {
               'h': {'class': 'logging.FileHandler',
                     'filename': 'test.log',
                     'formatter': 'normal',
                     'level': logging.INFO}},
           'loggers' : {
               '': {'handlers': ['h'], 'level': logging.DEBUG}}
         }