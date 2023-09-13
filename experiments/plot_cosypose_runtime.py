from pathlib import Path
import time
import numpy as np

from olt.localizer import Localizer
from olt.config import OBJ_MODEL_DIRS, LocalizerConfig
from olt.evaluation_tools import BOPDatasetReader

import matplotlib.pyplot as plt

SHOW = True
# FIG_EXT = 'pdf'
FIG_EXT = 'png'

PLOTS_DIR_NAME = Path('plots')
PLOTS_DIR_NAME.mkdir(exist_ok=True)

ds_name = 'ycbv'
# RENDERER_NAME = 'bullet'
RENDERER_NAME = 'panda'

localizer_cfg = LocalizerConfig()
localizer_cfg.n_workers = 0
localizer = Localizer(ds_name, localizer_cfg)



reader = BOPDatasetReader(ds_name, load_depth=True)
all_sids = sorted(reader.map_sids_vids.keys())
sid = all_sids[1]
vid = reader.map_sids_vids[sid][0]
K, height, width = reader.get_Kres(sid, vid)
obs = reader.get_obs(sid, vid)

# Warmup
poses, scores = localizer.predict(obs.rgb, K, n_coarse=1, n_refiner=1)





#####################################
#####################################
#####################################

N_run_predict = 50
n_refiner_lst = [1,2,3,4,5,6]
dt_lst_dic = {n_refiner: [] for n_refiner in n_refiner_lst}
for n_refiner in n_refiner_lst:
    for i in range(N_run_predict):
        vid = reader.map_sids_vids[sid][i]
        obs = reader.get_obs(sid, vid)

        print(f'{i}/{N_run_predict}')
        t = time.perf_counter()
        poses, scores = localizer.predict(obs.rgb, K, n_coarse=1, n_refiner=n_refiner)
        # print(poses)
        dt = 1000*(time.perf_counter() - t)
        dt_lst_dic[n_refiner].append(dt)



file_name = f'cosy_runtime_n_refiner_lst={n_refiner_lst}_renderer={RENDERER_NAME}_n_workers={localizer_cfg.n_workers}'
file_name += f'_{RENDERER_NAME}'
file_path = PLOTS_DIR_NAME / f'{file_name}.{FIG_EXT}'

plt.figure()
plt.title(f'CosyPose runtimes for multiple trials \n {file_name}')
for n_refiner in n_refiner_lst:
    plt.plot(np.arange(N_run_predict), dt_lst_dic[n_refiner], 'x', label=f'n_refiner={n_refiner}')
plt.xlabel('run #')
plt.ylabel('Runtime (ms)')
plt.grid()
plt.legend()
print('Saving ',file_path)
plt.savefig(file_path.as_posix())





#####################################
#####################################
#####################################

TCO_init, extra_data = localizer.get_cosy_predictions(obs.rgb, K, n_coarse=1, n_refiner=3, TCO_init=None, run_detector=True)

N_run_track = 50
n_refiner_lst = [1,2,3,4]
dt_lst_dic = {n_refiner: [] for n_refiner in n_refiner_lst}
for n_refiner in n_refiner_lst:
    for i in range(N_run_track):
        vid = reader.map_sids_vids[sid][i]
        obs = reader.get_obs(sid, vid)

        print(f'{i}/{N_run_track}')
        t = time.perf_counter()
        poses, scores = localizer.track(obs.rgb, K, TCO_init, n_refiner=n_refiner)
        # print(poses)
        dt = 1000*(time.perf_counter() - t)
        dt_lst_dic[n_refiner].append(dt)

file_name = f'cosy_track_dt_n_refiner_n_workers={localizer_cfg.n_workers}'
file_name += f'_{RENDERER_NAME}'
file_path = PLOTS_DIR_NAME / f'{file_name}.{FIG_EXT}'

plt.figure()
plt.title(f'CosyPose refiner runtimes for multiple trials \n {file_name}')
for n_refiner in n_refiner_lst:
    plt.plot(np.arange(N_run_track), dt_lst_dic[n_refiner], 'x', label=f'n_refiner={n_refiner}')
plt.xlabel('run #')
plt.ylabel('DT (ms)')
plt.grid()
plt.legend()
print('Saving ',file_path)
plt.savefig(file_path.as_posix())


if SHOW:
    plt.show()
