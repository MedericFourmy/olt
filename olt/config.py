import os
from pathlib import Path

DATASET_NAMES = ['ycbv']

MEGAPOSE_DATA_DIR = os.environ.get('MEGAPOSE_DATA_DIR')
assert(len(MEGAPOSE_DATA_DIR) > 0)
MEGAPOSE_DATA_DIR = Path(MEGAPOSE_DATA_DIR)

BOP_DS_DIRS = {ds_name: MEGAPOSE_DATA_DIR / Path('bop_datasets') / ds_name for ds_name in DATASET_NAMES}
OBJECT_MODEL_DIRS = {ds_name: MEGAPOSE_DATA_DIR / Path('urdfs') / ds_name for ds_name in DATASET_NAMES}
