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

    # RegionModality params (partial)
    region_scales: list[int] = default_list([7, 4, 2])
    region_standard_deviations: list[float] = default_list([25.0, 15.0, 10.0])

    # DepthModality params (partial)
    depth_considered_distances: list[float] = default_list([0.07, 0.05, 0.04])
    depth_standard_deviations: list[float] = default_list([0.05, 0.03, 0.02])


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