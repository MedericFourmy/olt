
import numpy as np
import quaternion
from pathlib import Path
import cv2
from PIL import Image



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


def create_video_from_images(img_dir: Path, output_name='out.mp4', ext: str ='.png', fps: int = 30.0):
    img_files = list(sorted(img_dir.glob(f'*{ext}')))
    if len(img_files) == 0:
        raise FileNotFoundError(f'{img_dir.as_posix()} directory does not contain {ext} files')

    frame1 = cv2.imread(img_files[0])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        filename=output_name, fourcc=fourcc, fps=fps, frameSize=frame1.shape
    )

    # Read each image and write it to the video
    for image in img_files:
        # read the image using OpenCV
        frame = cv2.imread(image)
        # Optional step to resize the input image to the dimension stated in the
        # VideoWriter object above
        # frame = cv2.resize(frame, dsize=(430, 430))
        video.write(frame)
    
    # Exit the video writer
    video.release()



if __name__ == '__main__':
    img_dir = Path('../tmp/imgs/')
    create_video_from_images(img_dir)