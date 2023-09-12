import datetime
# from olt.localizer import Localizer
from olt.evaluation_tools import BOPDatasetReader, append_result, run_bop_evaluation

reader = BOPDatasetReader('ycbv', load_depth=False)

FREQ = 30.0

nb_total_views = sum(len(vids) for vids in reader.map_sids_vids.values())


print(str(datetime.timedelta(seconds=nb_total_views/FREQ)))