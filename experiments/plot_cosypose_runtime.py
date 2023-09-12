from pathlib import Path
import time
import numpy as np
import cv2

from olt.localizer import Localizer
from olt.config import OBJ_MODEL_DIRS, LocalizerConfig
from olt.evaluation_tools import BOPDatasetReader

import matplotlib.pyplot as plt

SHOW = True
FIG_EXT = 'pdf'

PLOTS_DIR_NAME = Path('plots')
PLOTS_DIR_NAME.mkdir(exist_ok=True)

ds_name = 'ycbv'
RENDERER_NAME = 'bullet'
# RENDERER_NAME = 'panda'

localizer_cfg = LocalizerConfig()
localizer_cfg.n_workers = 0
localizer = Localizer(ds_name, localizer_cfg)

N_REFINER = 6


reader = BOPDatasetReader(ds_name, load_depth=True)
all_sids = sorted(reader.map_sids_vids.keys())
sid = all_sids[1]
vid = reader.map_sids_vids[sid][0]
K, height, width = reader.get_Kres(sid, vid)
obs = reader.get_obs(sid, vid)

# Warmup
poses, scores = localizer.predict(obs.rgb, K, n_coarse=1, n_refiner=N_REFINER)





#####################################
#####################################
#####################################

# N_run_predict = 20
# dt_lst = []
# for i in range(N_run_predict):
#     vid = reader.map_sids_vids[sid][i]
#     obs = reader.get_obs(sid, vid)

#     print(f'{i}/{N_run_predict}')
#     t = time.perf_counter()
#     poses, scores = localizer.predict(obs.rgb, K, n_coarse=1, n_refiner=N_REFINER)
#     # print(poses)
#     dt = 1000*(time.perf_counter() - t)
#     dt_lst.append(dt)



# file_name = f'cosy_dt_n_refiner={N_REFINER}_n_workers={localizer_cfg.n_workers}'
# file_name += f'_{RENDERER_NAME}'
# file_path = PLOTS_DIR_NAME / f'{file_name}.{FIG_EXT}'

# plt.figure()
# plt.title(f'Running CosyPose multiple times \n {file_name}')
# plt.plot(np.arange(N_run_predict), dt_lst, 'x')
# plt.xlabel('run #')
# plt.ylabel('DT (ms)')
# print('Saving ',file_path)
# plt.savefig(file_path.as_posix())





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
plt.title(f'Running CosyPose Refiner \n {file_name}')
for n_refiner in n_refiner_lst:
    plt.plot(np.arange(N_run_track), dt_lst_dic[n_refiner], 'x', label=f'n_refiner={n_refiner}')
plt.xlabel('run #')
plt.ylabel('DT (ms)')
plt.savefig(f'{file_name}.{FIG_EXT}')
print('Saving ',file_path)
plt.savefig(file_path.as_posix())


if SHOW:
    plt.show()
