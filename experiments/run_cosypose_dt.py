
import time
import numpy as np

from olt.localizer import Localizer
from olt.config import OBJ_MODEL_DIRS, LocalizerConfig
from olt.evaluation_tools import BOPDatasetReader

import matplotlib.pyplot as plt

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
poses = localizer.predict(obs.rgb, K, n_coarse=1, n_refiner=N_REFINER)


N_run = 2000
dt_lst = []
for i in range(N_run):
    vid = reader.map_sids_vids[sid][i]
    obs = reader.get_obs(sid, vid)

    print(f'{i}/{N_run}')
    t = time.perf_counter()
    poses = localizer.predict(obs.rgb, K, n_coarse=1, n_refiner=N_REFINER)
    # print(poses)
    dt = 1000*(time.perf_counter() - t)
    dt_lst.append(dt)



file_name = f'cosy_dt_n_refiner={N_REFINER}_n_workers={localizer_cfg.n_workers}'
file_name += f'_{RENDERER_NAME}'

plt.figure()
plt.title(f'Running CosyPose multiple times \n {file_name}')
plt.plot(np.arange(N_run), dt_lst, 'x')
plt.xlabel('run #')
plt.ylabel('DT (ms)')
plt.savefig(file_name)
plt.show()
