
import numpy as np
import quaternion


def tq_to_SE3(t, q):
    """
    t: translation as list or array
    q: quaternion as list or array, expected order: xyzw
    out: 4x4 array representing the SE(3) transformation
    """
    T = np.eye(4)
    T[:3,3] = t
    # np.quaternion constructor uses wxyz order convention
    quat = np.quaternion(q[3], q[0], q[1], q[2]).normalized()
    T[:3,:3] = quaternion.as_rotation_matrix(quat)
    return T


def intrinsics2Kres(fu, fv, ppu, ppv, width, height):
    return np.array([
        fu, 0, ppu,
        0, fv, ppv,
        0, 0, 1,
    ]).reshape((3,3)), width, height


def Kres2intrinsics(K, width, height):
    return {
        'fu': K[0,0], 
        'fv': K[1,1], 
        'ppu': K[0,2], 
        'ppv': K[1,2],
        'width': width,
        'height': height, 
    }