import numpy as np

from olt.evaluation_tools import append_result



def test_append_results():
    from bop_toolkit_lib import inout  # noqa

    preds = [] 
    scene_id = 1 
    obj_id = 3
    view_id = 0 
    score = 0.6
    TCO = np.eye(4) 
    dt = 1.0

    append_result(preds, scene_id, obj_id, view_id, score, TCO, dt)
    view_id+=1
    append_result(preds, scene_id, obj_id, view_id, score, TCO, dt)
    view_id+=1
    append_result(preds, scene_id, obj_id, view_id, score, TCO, dt)
    view_id+=1

    dummy_res = 'results_test.csv'
    inout.save_bop_results(dummy_res, preds)

    with open(dummy_res, 'r') as f:
        assert len(f.readlines()) == 4
