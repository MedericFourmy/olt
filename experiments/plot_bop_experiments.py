import json
from pathlib import Path
import matplotlib.pyplot as plt
from olt.utils import get_method_name
import seaborn as sns
plt.rcParams['pdf.fonttype'] = 42  # avoid pbe with fonts when submitting paperplaza

# colors = sns.color_palette("deep")
colors = sns.color_palette("colorblind")

FILE_EXTS = ['png', 'pdf']
EVALUATIONS_DIR_NAME = Path('evaluations')
PLOTS_DIR_NAME = Path('plots')
PLOTS_DIR_NAME.mkdir(exist_ok=True)

ds_name = 'ycbv'



def get_scores(run_name):
    result_bop_eval_dir = EVALUATIONS_DIR_NAME / f'{run_name}_{ds_name}-test'
    if not result_bop_eval_dir.exists():
        print('Missing eval:', result_bop_eval_dir)
        return None

    scores19_path = result_bop_eval_dir / 'scores_bop19.json'

    with open(scores19_path.as_posix(), 'r') as fp:
        return  json.load(fp)



FREQS = [5,10,15,20,30,60,90,120]


training_type = 'synt+real'
renderer_name = 'bullet'
# modalities = ['rgb']
modalities = ['rgb', 'rgbd']

methods2labels = {
    'threaded': 'OLT (ours)',
    'threaded-notrack': 'OLT-NoTracker',
    'trackfromstart': 'Tracker-InitLocalizer',
    'cosyrefined': 'LocalizerRefinedWithTracker',
    'cosyonly': 'Localizer',
}

colors = {
    'threaded': colors[2],  # green
    'threaded-notrack': colors[1],  # orange
    'trackfromstart': colors[0],  # blue
    'cosyrefined': colors[7],  # black
    # 'cosyonly': colors[3],  # red
    'cosyonly': 'tab:red',  # red
}

linestyles = {
    'threaded': '-',
    'threaded-notrack': '-',
    'trackfromstart': '-',
    'cosyrefined': '-.',
    'cosyonly': '--',
}

MARKERSIZE = 10
linewidth = 3.0



def plot_ar_freq(modality):

    fig = plt.figure(f'AR=f(freq) {modality}', figsize=(6.4*1.4,4.8))

    # # FORMER NOT SO GOOD RESULTS
    # # Continuous implementation
    # method = 'threaded'
    # ar_lst = []
    # available_freqs = []
    # for freq in FREQS:

    #     run_name = get_method_name(method, 
    #                                 training_type,
    #                                 renderer_name,
    #                                 f'{freq}Hz',
    #                                 modality)


    #     scores19 = get_scores(run_name)
    #     if scores19 is None: continue

    #     ar = scores19['bop19_average_recall']
    #     ar_lst.append(ar)
    #     available_freqs.append(freq)

    # plt.plot(available_freqs, ar_lst, color='g', linestyle=linestyles[method], marker='o', markersize=MARKERSIZE, label=f'{methods2labels[method]}+prev')



    # NEW HIGH TIKHONOV REGULARIZATION RESULT
    method = 'threaded'
    n_refiner = 2
    ntrackit = 2
    ar_lst = []
    available_freqs = []
    for freq in FREQS:

        run_name = get_method_name(method, 
                            training_type,
                            renderer_name,
                            f'{freq}Hz',
                            f'nr{n_refiner}',
                            modality,
                            f'ntrackit{ntrackit}',
                            'tikhohigh'
                            )


        scores19 = get_scores(run_name)
        if scores19 is None: continue

        ar = scores19['bop19_average_recall']
        ar_lst.append(ar)
        available_freqs.append(freq)

    plt.plot(available_freqs, ar_lst, color=colors[method], linewidth=linewidth, linestyle=linestyles[method], marker='o', markersize=MARKERSIZE, label=f'{methods2labels[method]}')


    # Localizer WITH DELAY
    method = 'threaded'
    n_refiner = 2
    ntrackit = 0
    ar_lst = []
    available_freqs = []
    for freq in FREQS:
        
        run_name = get_method_name(method, 
                            training_type,
                            renderer_name,
                            f'{freq}Hz',
                            f'nr{n_refiner}',
                            modality,
                            f'ntrackit{ntrackit}',
                            'tikhohigh'
                            )

        scores19 = get_scores(run_name)
        if scores19 is None: continue

        ar = scores19['bop19_average_recall']
        ar_lst.append(ar)
        available_freqs.append(freq)

    method += '-notrack'
    plt.plot(available_freqs, ar_lst, color=colors[method], linewidth=linewidth, linestyle=linestyles[method], marker='o', markersize=MARKERSIZE, label=methods2labels[method])


    # method = 'threaded'
    # n_refiner = 2
    # ar_lst = []
    # available_freqs = []
    # for freq in FREQS:

    #     run_name = get_method_name(method, 
    #                         training_type,
    #                         renderer_name,
    #                         f'{freq}Hz',
    #                         f'nr{n_refiner}',
    #                         modality,
    #                         'tikohigh'
    #                         )

    #     scores19 = get_scores(run_name)
    #     if scores19 is None: continue

    #     ar = scores19['bop19_average_recall']
    #     ar_lst.append(ar)
    #     available_freqs.append(freq)
        
    # plt.plot(available_freqs, ar_lst, color='y', linestyle=linestyles[method], marker='o', markersize=MARKERSIZE, label=f'{method}-tikohigh-ntrackit{ntrackit}')


    method = 'trackfromstart'
    # for use_gt_for_localization in [False, True]:
    for use_gt_for_localization in [False]:
        run_name = get_method_name(method,
                                    f'from-gt={use_gt_for_localization}',
                                    modality
                                    )
        scores19 = get_scores(run_name)
        if scores19 is not None:
            gt_label = 'gt' if use_gt_for_localization else 'cosy'
            plt.hlines(scores19['bop19_average_recall'], 0, FREQS[-1], color=colors[method], linewidth=linewidth, linestyle=linestyles[method], label=f'{methods2labels[method]}')

    # method = 'cosyrefined'
    # run_name = get_method_name(method, 
    #                             training_type,
    #                             renderer_name,
    #                             modality,
    #                             )
    # scores19 = get_scores(run_name)
    # if scores19 is not None:
    #     plt.hlines(scores19['bop19_average_recall'], 0, FREQS[-1], color=colors[method], linestyle=linestyles[method], label=f'{methods2labels[method]}')

    method = 'cosyonly'
    # n_refiner_lst = [1,2,4]
    n_refiner_lst = [4]
    for i, n_refiner in enumerate(n_refiner_lst):
        run_name = get_method_name(method, 
                                    training_type,
                                    renderer_name,
                                    f'nr{n_refiner}')
        scores19 = get_scores(run_name)
        plt.hlines(scores19['bop19_average_recall'], 0, FREQS[-1], color=colors[method], linewidth=linewidth, linestyles=linestyles[method], label=f'Localizer')


    ######################################
    ######################################
    # Formatting
    ######################################
    ######################################
    # font = {'fontname':'Comic Sans MS'}
    font = {'fontname':'serif'}
    
    plt.xlabel('Images rate (Hz)',   fontsize=18, **font)
    plt.ylabel('BOP Average Recall', fontsize=18, **font)
    plt.ylim(0.3,1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=2, fancybox=True, shadow=True, fontsize=18)
    fig.subplots_adjust(bottom=0.13)
    # fig.subplots_adjust(top=0.13)
    
    plot_paths = [PLOTS_DIR_NAME / f'ar_f_freq_{modality}.{ext}' for ext in FILE_EXTS]
    for plot_path in plot_paths:
        print('Saving', plot_path)
        plt.savefig(plot_path.as_posix())




for modality in sorted(modalities, reverse=True):
    plot_ar_freq(modality=modality)





# plt.figure('Cosyonly multiple n_refiners')
# method = 'cosyonly'
# n_refiner_lst = [1,2,3,4,5,6]
# ar_lst = []
# for n_refiner in n_refiner_lst:
#     run_name = get_method_name(method, 
#                                 training_type,
#                                 renderer_name,
#                                 f'nr{n_refiner}')
#     scores19 = get_scores(run_name)
#     ar = scores19['bop19_average_recall']
#     ar_lst.append(ar)
# plt.plot(n_refiner_lst, ar_lst, 'x')
# plt.xlabel('# refinfer steps')
# plt.ylabel('AR score BOP')
# plt.ylim(0.7,0.9)






plt.show()
    


