import os
from pathlib import Path
from dataclasses import dataclass

DATASET_NAMES = ['ycbv']

MEGAPOSE_DATA_DIR = os.environ.get('MEGAPOSE_DATA_DIR')
assert(len(MEGAPOSE_DATA_DIR) > 0)
MEGAPOSE_DATA_DIR = Path(MEGAPOSE_DATA_DIR)

BOP_DS_DIRS = {ds_name: MEGAPOSE_DATA_DIR / Path('bop_datasets') / ds_name for ds_name in DATASET_NAMES}
OBJECT_MODEL_DIRS = {ds_name: MEGAPOSE_DATA_DIR / Path('urdfs') / ds_name for ds_name in DATASET_NAMES}


@dataclass
class TrackerConfig:
    tmp_dir_name: str = 'tmp'
    viewer_name: str = 'normal_viewer'
    n_corr_iterations: int = 3
    n_update_iterations: int = 5


@dataclass
class LocalizerConfig:
    detector_threshold: float = 0.8


@dataclass
class EvaluationBOPConfig:
    tracker_cfg: TrackerConfig = TrackerConfig()
    localizer_cfg: LocalizerConfig = LocalizerConfig()
    ds_name: str = 'ycbv'
    nb_imgs_load: int = -1