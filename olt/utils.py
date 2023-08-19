
import numpy as np
import quaternion
from pathlib import Path
import cv2
from PIL import Image
import resource
get_mem_usage = lambda : f'Memory usage: {int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000} (Mb)'
print_mem_usage = lambda : print(get_mem_usage())



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


def create_video_from_images(img_dir: Path, vid_name='out.mp4', ext: str ='.png', fps: int = 30.0):
    # img_files = list(sorted(img_dir.glob(f'*{ext}')))
    img_files = list(img_dir.glob(f'*{ext}'))

    # img file name format is unfortunate: normal_viewer_image_18 -> not lexicographic
    save_ids = [int(name.stem.split('_')[-1]) for name in img_files]

    img_files = [name for id, name in sorted(zip(save_ids, img_files))]


    if len(img_files) == 0:
        raise FileNotFoundError(f'{img_dir.as_posix()} directory does not contain {ext} files')

    # Get dimensions of of the 
    h, w, _ = cv2.imread(img_files[0].as_posix()).shape

    output_path = img_dir / vid_name
    video = cv2.VideoWriter(filename=output_path.as_posix(), 
                            fourcc=cv2.VideoWriter_fourcc(*"mp4v"), 
                            fps=fps, frameSize=(w, h)
    )

    for img_path in img_files:
        frame = cv2.imread(img_path.as_posix())
        video.write(frame)
    
    video.release()
    print(f'Written {output_path}')

def obj_id2label(id):
    return f'obj_{id:06}'


if __name__ == '__main__':
    img_dir = Path('../tmp/imgs/')
    create_video_from_images(img_dir, fps=30)

    # print(obj_id2label(48))