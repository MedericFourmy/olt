import datetime
from olt.evaluation_tools import BOPDatasetReader, append_result, run_bop_evaluation

reader = BOPDatasetReader('ycbv', load_depth=False)

FREQS = [2,5,10,15,20,30,60,90,120]

nb_total_views = sum(len(vids) for vids in reader.map_sids_vids.values())

print()
print('Estimated experiment durations: h:m:s')
for f in FREQS:
    print(f'{f} Hz', str(datetime.timedelta(seconds=nb_total_views/f)))





# For ycbv:

# Estimated experiment durations: h:m:s
# 2 Hz 2:52:49
# 5 Hz 1:09:07.600000
# 10 Hz 0:34:33.800000
# 15 Hz 0:23:02.533333
# 20 Hz 0:17:16.900000
# 30 Hz 0:11:31.266667
# 60 Hz 0:05:45.633333
# 90 Hz 0:03:50.422222
# 120 Hz 0:02:52.816667