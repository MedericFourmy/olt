import json
from pathlib import Path
import matplotlib.pyplot as plt
from olt.utils import get_method_name

EVALUATIONS_DIR_NAME = Path('evaluations')
PLOTS_DIR_NAME = Path('plots')
PLOTS_DIR_NAME.mkdir(exist_ok=True)

ds_name = 'ycbv'


# FREQS = [2,4,8,16,32,64,128]
FREQS = [5,10,15,20,30,60]
# FREQS = [None]

plt.figure('AR=f(freq)')
plt.title('AR=f(freq)')

training_type = 'synt+real'
renderer_name = 'bullet'
modalities = ['rgb', 'rgbd']

colors = {
    'rgb': 'b',
    'rgbd': 'c'
}

for modality in modalities:
    ar_lst = []
    available_freqs = []
    for freq in FREQS:

        method_name = get_method_name('threaded', 
                                    training_type,
                                    renderer_name,
                                    f'{freq}Hz',
                                    modality)

        result_bop_eval_dir = EVALUATIONS_DIR_NAME / f'{method_name}_{ds_name}-test'
        if not result_bop_eval_dir.exists():
            print('Missing eval:', result_bop_eval_dir)
            continue

        scores19_path = result_bop_eval_dir / 'scores_bop19.json'
        print(scores19_path)

        with open(scores19_path.as_posix(), 'r') as fp:
            scores = json.load(fp)

        ar = scores['bop19_average_recall']
        ar_lst.append(ar)
        available_freqs.append(freq)

    
    c = colors[modality]
    plt.plot(available_freqs, ar_lst, f'{c}x', label=f'conti-{modality}')

plt.legend()
plot_path = PLOTS_DIR_NAME / 'ar_f_freq.png'
plt.savefig(plot_path.as_posix())
plt.show()
    


