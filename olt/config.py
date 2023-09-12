import os
from pathlib import Path
from dataclasses import dataclass, field
import logging

DATASET_NAMES = ['ycbv', 'rotd']

MEGAPOSE_DATA_DIR = os.environ.get('MEGAPOSE_DATA_DIR')
assert len(MEGAPOSE_DATA_DIR) > 0, 'MEGAPOSE_DATA_DIR env variable is not set'
MEGAPOSE_DATA_DIR = Path(MEGAPOSE_DATA_DIR)

BOP_DS_DIRS = {ds_name: MEGAPOSE_DATA_DIR / Path('bop_datasets') / ds_name for ds_name in DATASET_NAMES}
OBJ_MODEL_DIRS = {ds_name: MEGAPOSE_DATA_DIR / Path('urdfs') / ds_name for ds_name in DATASET_NAMES}

# Adapted from https://stackoverflow.com/questions/53632152/why-cant-dataclasses-have-mutable-defaults-in-their-class-attributes-declaratio
default_list = lambda l: field(default_factory=lambda: l)



@dataclass
class ModelConfig:
    sphere_radius: float = 0.8
    n_divides: int = 4 
    n_points: int = 200
    max_radius_depth_offset: float = 0.05
    stride_depth_offset: float = 0.002 
    use_random_seed: bool = False
    image_size: int = 2000


@dataclass
class RegionModalityConfig:

    # Parameters for general distribution
    n_lines_: int = 200
    min_continuous_distance_: float = 3.0
    function_length_: int = 8
    distribution_length_: int = 12
    function_amplitude_: float = 0.43
    function_slope_: float = 0.5
    learning_rate_: float = 1.3
    n_global_iterations_: int = 1
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
    n_points: int = 200
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
Default value taken from ICG YCBV evaluation
"""
@dataclass
class TrackerConfig:
    tmp_dir_name: str = 'tmp'
    viewer_name: str = 'normal_viewer'
    viewer_display: bool = False
    viewer_save: bool = False
    use_depth: bool = False
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