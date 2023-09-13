import json
from pathlib import Path
import matplotlib.pyplot as plt
from olt.utils import get_method_name

FILE_EXT = 'png'
# FILE_EXT = 'pdf'
EVALUATIONS_DIR_NAME = Path('evaluations')
PLOTS_DIR_NAME = Path('plots')
PLOTS_DIR_NAME.mkdir(exist_ok=True)

ds_name = 'ycbv'


# FREQS = [2,4,8,16,32,64,128]
FREQS = [5,10,15,20,30,60,90]
# FREQS = [None]

plt.figure('AR=f(freq)')
# plt.title('AR=f(freq)')

training_type = 'synt+real'
renderer_name = 'bullet'
modalities = ['rgb', 'rgbd']

methods2labels = {
    'threaded': 'olt',
    'cosyonly': 'cosypose',
    'trackfromstart': 'ICG-track',
    'cosyrefined': 'cosy+ICG',
}

colors = {
    'threaded': 'b',
    'cosyonly': 'c',
    'trackfromstart': 'g',
    'cosyrefined': 'k',
}

linestyles = {
    'rgb': '-',
    'rgbd': '--'
}

def get_scores(run_name):
    result_bop_eval_dir = EVALUATIONS_DIR_NAME / f'{run_name}_{ds_name}-test'
    if not result_bop_eval_dir.exists():
        print('Missing eval:', result_bop_eval_dir)
        return None

    scores19_path = result_bop_eval_dir / 'scores_bop19.json'

    with open(scores19_path.as_posix(), 'r') as fp:
        return  json.load(fp)

# Continuous implementation
method = 'threaded'
for modality in modalities:
    ar_lst = []
    available_freqs = []
    for freq in FREQS:

        run_name = get_method_name(method, 
                                    training_type,
                                    renderer_name,
                                    f'{freq}Hz',
                                    modality)


        scores19 = get_scores(run_name)
        if scores19 is None: continue

        ar = scores19['bop19_average_recall']
        ar_lst.append(ar)
        available_freqs.append(freq)

    
    plt.plot(available_freqs, ar_lst, color=colors[method], linestyle=linestyles[modality], marker='o', markersize=5, label=f'{methods2labels[method]}-{modality}')

method = 'threaded'
n_refiner = 2
for modality in modalities:
    ar_lst = []
    available_freqs = []
    for freq in FREQS:

        run_name = get_method_name(method, 
                                   training_type,
                                   renderer_name,
                                   f'{freq}Hz',
                                   modality,
                                   'tikohigh',
                                   f'nr2',
                                   )
        run_name = get_method_name(method, 
                            training_type,
                            renderer_name,
                            f'{freq}Hz',
                            f'nr{n_refiner}',
                            modality,
                            'tikohigh'
                            )


        scores19 = get_scores(run_name)
        if scores19 is None: continue

        ar = scores19['bop19_average_recall']
        ar_lst.append(ar)
        available_freqs.append(freq)

    
    plt.plot(available_freqs, ar_lst, color='g', linestyle=linestyles[modality], marker='o', markersize=5, label=f'{method+"-tikohigh"}-{modality}')



method = 'threaded'
for modality in modalities:
    ar_lst = []
    available_freqs = []
    for freq in FREQS:

        run_name = get_method_name(method, 
                                   training_type,
                                   renderer_name,
                                   f'{freq}Hz',
                                   modality,
                                   'ter')


        scores19 = get_scores(run_name)
        if scores19 is None: continue

        ar = scores19['bop19_average_recall']
        ar_lst.append(ar)
        available_freqs.append(freq)

    
    plt.plot(available_freqs, ar_lst, color='orange', linestyle=linestyles[modality], marker='o', markersize=5, label=f'{method+"-ter"}-{modality}')


method = 'threaded'
for modality in modalities:
    ar_lst = []
    available_freqs = []
    for freq in FREQS:

        run_name = get_method_name(method, 
                                   training_type,
                                   renderer_name,
                                   f'{freq}Hz',
                                   modality,
                                   '4th')


        scores19 = get_scores(run_name)
        if scores19 is None: continue

        ar = scores19['bop19_average_recall']
        ar_lst.append(ar)
        available_freqs.append(freq)

    
    plt.plot(available_freqs, ar_lst, color='k', linestyle=linestyles[modality], marker='o', markersize=5, label=f'{method+"-4th"}-{modality}')



method = 'trackfromstart'
for modality in modalities:
    for use_gt_for_localization in [False, True]:
        run_name = get_method_name(method,
                                   f'from-gt={use_gt_for_localization}',
                                    modality
                                    )
        print(run_name)
        scores19 = get_scores(run_name)
        if scores19 is not None:
            gt_label = 'gt' if use_gt_for_localization else 'cosy'
            plt.hlines(scores19['bop19_average_recall'], 0, FREQS[-1], color=colors[method], linestyle=linestyles[modality], label=f'{methods2labels[method]}-{gt_label}-{modality}')

method = 'cosyrefined'
for modality in modalities:
    run_name = get_method_name(method, 
                                training_type,
                                renderer_name,
                                modality,
                                )
    scores19 = get_scores(run_name)
    if scores19 is not None:
        plt.hlines(scores19['bop19_average_recall'], 0, FREQS[-1], color=colors[method], linestyle=linestyles[modality], label=f'{methods2labels[method]}-{modality}')


plt.xlabel('Playback rate (Hz)')
plt.ylabel('BOP Average Recall')
plt.ylim(0.0,1)
plt.grid()

# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#        ncol=2, mode="expand", borderaxespad=0.)
plt.legend(loc='lower center', ncol=2)
plot_path = PLOTS_DIR_NAME / f'ar_f_freq.{FILE_EXT}'
plt.savefig(plot_path.as_posix())





plt.figure('Cosyonly multiple n_refiners')
method = 'cosyonly'
n_refiner_lst = [1,2,3,4,5,6]
ar_lst = []
for n_refiner in n_refiner_lst:
    run_name = get_method_name(method, 
                                training_type,
                                renderer_name,
                                f'nr{n_refiner}')
    scores19 = get_scores(run_name)
    ar = scores19['bop19_average_recall']
    ar_lst.append(ar)
plt.plot(n_refiner_lst, ar_lst, 'x')
plt.xlabel('# refinfer steps')
plt.ylabel('AR score BOP')
plt.ylim(0.7,0.9)






plt.show()
    


