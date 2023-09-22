# Happy rendering
import os
import numpy as np
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData
from happypose.toolbox.utils.conversion import convert_scene_observation_to_panda3d
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer import Panda3dLightData
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.toolbox.visualization.bokeh_plotter import BokehPlotter
from happypose.toolbox.visualization.utils import make_contour_overlay
from bokeh.io import export_png
from bokeh.plotting import gridplot

from olt.config import OBJ_MODEL_DIRS



def make_object_dataset(object_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    object_dirs = (object_dir).iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
        # TODO: fix mesh units
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def render_overlays(ds_name, rgb_lst, K, height, width, object_labels_lst, T_co_lst_per_img, colors_lst, vis_dir, prefix='contour_overlay'):

    assert len(rgb_lst) == len(object_labels_lst) == len(T_co_lst_per_img) == len(colors_lst)

    object_dataset = make_object_dataset(OBJ_MODEL_DIRS[ds_name])

    camera_data = CameraData(K, (height, width))
    camera_data.TWC = Transform(np.eye(4))

    N_img = len(rgb_lst)
    renderer = Panda3dSceneRenderer(object_dataset)

    for i in range(N_img):
        print(f'Image {i}/{N_img}')

        object_labels = object_labels_lst[i]
        T_co_lst = T_co_lst_per_img[i]
        colors = colors_lst[i]
        rgb = rgb_lst[i]

        assert len(object_labels) == len(T_co_lst) == len(colors)

        contour_overlays = []
        for j, (target_label, T_co, color) in enumerate(zip(object_labels, T_co_lst, colors)):
            
            object_datas = [
                ObjectData(label=target_label, TWO=Transform(T_co))
            ] 
                
            camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
            light_datas = [
                Panda3dLightData(
                    light_type="ambient",
                    color=((1.0, 1.0, 1.0, 1)),
                ),
            ]
            # Should be possible to render virtual cameras at the same time
            # would permit to properly batch render 
            renderings = renderer.render_scene(
                object_datas,
                [camera_data],
                light_datas,
                render_depth=False,
                render_binary_mask=False,
                render_normals=False,
                copy_arrays=True,
            )[0]

            plotter = BokehPlotter()

            # fig_rgb = plotter.plot_image(rgb)
            # fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
            contour_overlay = make_contour_overlay(
                rgb, renderings.rgb, dilate_iterations=3, color=color
            )["img"]

            contour_overlays.append(contour_overlay)

        # Average out the images where 1 contour is printed
        ratio = 1/len(contour_overlays)
        contour_overlay_aggr = sum(ratio*overlay for overlay in contour_overlays) 

        contour_overlay_aggr = contour_overlay_aggr.astype(np.uint8)

        fig_contour_overlay = plotter.plot_image(contour_overlay_aggr)
        export_png(fig_contour_overlay, filename=vis_dir / f"{prefix}_{i:06}.png")



if __name__ == '__main__':

    import pinocchio as pin

    ds_name = 'ycbv'

    vis_dir = Path("visualizations")
    vis_dir.mkdir(exist_ok=True)

    # inputs
    fx, fy, cx, cy = 382, 382, 322, 249
    K = np.array([
        fx, 0, cx,
        0, fy, cy,
        0,  0,  1
    ]).reshape((3,3))
    height, width = 480, 640
    n_img = 2

    rgb_lst = [np.ones((height, width, 3)).astype(np.uint8) for _ in range(n_img)]

    object_labels_lst = [
        ['obj_000010', 'obj_000008'],
        ['obj_000012', 'obj_000013', 'obj_000004']
    ]

    colors_lst = [
        [(0, 255, 0), (0, 255, 255)],
        [(0, 255, 0), (0, 255, 255), (255, 255, 0)]
    ]
    
    T_1_10 = pin.XYZQUATToSE3([0.1, 0.05, 0.65,    -0.5, -0.5, 0.5, -0.5 ])
    T_1_8 = pin.XYZQUATToSE3 ([0.2, 0.05, 0.65,    -0.5, -0.5, 0.5, -0.5 ])
    T_2_12 = pin.XYZQUATToSE3([0.0, 0.1, 0.65,    -0.5, -0.5, 0.5, -0.5 ])
    T_2_13 = pin.XYZQUATToSE3([0.0, 0.2, 0.65,    -0.5, -0.5, 0.5, -0.5 ])
    T_2_4 = pin.XYZQUATToSE3 ([0.0, 0.3, 0.65,    -0.5, -0.5, 0.5, -0.5 ])

    T_co_lst_per_img = [
        [T_1_10, T_1_8],
        [T_2_12, T_2_13, T_2_4],
    ]

    render_overlays(ds_name, rgb_lst, K, height, width, object_labels_lst, T_co_lst_per_img, colors_lst, vis_dir)