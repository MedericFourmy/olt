import datetime
from olt.evaluation_tools import BOPDatasetReader, append_result, run_bop_evaluation

reader = BOPDatasetReader('ycbv', load_depth=False)

FREQS = [2,5,10,15,20,30,60,90,120]

nb_total_views = sum(len(vids) for vids in reader.map_sids_vids.values())

print()
print('Estimated experiment durations: h:m:s')
for f in FREQS:
    print(f'{f} Hz', str(datetime.timedelta(seconds=nb_total_views/f)))