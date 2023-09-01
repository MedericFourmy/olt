import os
from pathlib import Path
from dataclasses import dataclass
import logging

DATASET_NAMES = ['ycbv', 'rotd']

MEGAPOSE_DATA_DIR = os.environ.get('MEGAPOSE_DATA_DIR')
assert len(MEGAPOSE_DATA_DIR) > 0, 'MEGAPOSE_DATA_DIR env variable is not set'
MEGAPOSE_DATA_DIR = Path(MEGAPOSE_DATA_DIR)

BOP_DS_DIRS = {ds_name: MEGAPOSE_DATA_DIR / Path('bop_datasets') / ds_name for ds_name in DATASET_NAMES}
OBJ_MODEL_DIRS = {ds_name: MEGAPOSE_DATA_DIR / Path('urdfs') / ds_name for ds_name in DATASET_NAMES}


@dataclass
class TrackerConfig:
    tmp_dir_name: str = 'tmp'
    viewer_name: str = 'normal_viewer'
    n_corr_iterations: int = 3
    n_update_iterations: int = 5
    viewer_display: bool = False
    viewer_save: bool = False
    tikhonov_parameter_rotation: float = 1000.0
    tikhonov_parameter_translation: float = 30000.0


@dataclass
class LocalizerConfig:
    detector_threshold: float = 0.8
    n_workers: int = 2


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