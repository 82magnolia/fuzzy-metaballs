import pytorch3d
import argparse
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
from pytorch3d.io import load_objs_as_meshes, load_ply
import torch
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    AmbientLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
from pytorch3d.io.ply_io import MeshPlyFormat, PointcloudPlyFormat
from pytorch3d.io.obj_io import MeshObjFormat
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
import os
from pathlib import Path
import warnings
import tkinter as tk
from PIL import Image, ImageTk

class PyPathManager:
    def __init__(self):
        pass
    def open(f, mode):
        return open(f, mode)
    def exists(f):
        return os.path.exists(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", help="Type of data to load", default="pcd")
    parser.add_argument("--file_name", help="Name of file to load", required=True)
    parser.add_argument("--save_dir", help="Directory for saving poses", default="./log")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    log_dir = args.save_dir
    img_dir = os.path.join(args.save_dir, 'images')
    sil_dir = os.path.join(args.save_dir, 'sil_images')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(sil_dir):
        os.makedirs(sil_dir)

    # Prepare camera
    R = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], device=device).unsqueeze(0)
    T = torch.zeros(3, device=device).unsqueeze(0)
    K = None

    cameras = FoVPerspectiveCameras(R=R, T=T, K=K, device=device)
    cam_intrinsics = cameras.compute_projection_matrix(cameras.znear, cameras.zfar, cameras.fov, \
                                                       cameras.aspect_ratio, cameras.degrees).cpu().numpy()

    # Extract screen space intrinsics for use in DROID-SLAM coordinate frame
    screen_cam_intrinsics = np.zeros_like(cam_intrinsics[0][:3, :3])

    img_h, img_w = 512, 512
    ndc_focal_length = cam_intrinsics[0][0, 0]
    screen_focal_length = ndc_focal_length * min(img_h, img_w) / 2
    screen_px = img_w / 2 - cam_intrinsics[0][0, 2] * min(img_h, img_w) / 2
    screen_py = img_h / 2 - cam_intrinsics[0][1, 2] * min(img_h, img_w) / 2
    screen_cam_intrinsics[0, 0] = screen_focal_length
    screen_cam_intrinsics[1, 1] = screen_focal_length
    screen_cam_intrinsics[0, 2] = screen_px
    screen_cam_intrinsics[1, 2] = screen_py
    screen_cam_intrinsics[2, 2] = 1.

    # Save intrinsics for use in DROID-SLAM
    with open(os.path.join(log_dir, 'droid_slam_intrinsics.txt'), 'w') as f:
        f.write(f"{screen_cam_intrinsics[0, 0]} {screen_cam_intrinsics[1, 1]} {screen_cam_intrinsics[0, 2]} {screen_cam_intrinsics[1, 2]}")

    # Prepare mesh or point cloud data
    if args.data_type == 'mesh':
        ext = args.file_name.split('.')[-1]
        print("Loading mesh...")
        if ext == 'ply':
            mesh_ply_reader = MeshPlyFormat()
            mesh = mesh_ply_reader.read(args.file_name, include_textures=True, device=device, path_manager=PyPathManager)
        elif ext == 'obj':
            warnings.warn(".obj with multiple texture files is not fully supported")
            mesh = load_objs_as_meshes([args.file_name])
        else:
            raise NotImplementedError("Other mesh data files not supported")
        print("Done!")

        mesh = mesh.to(device)

        raster_settings = RasterizationSettings(
            image_size=512, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        lights = AmbientLights(device=device)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=cameras,
                lights=lights
            )
        )

        model_3d = mesh

    elif args.data_type == 'pcd':
        ext = args.file_name.split('.')[-1]
        print("Loading point cloud (assume for .txt, .xyz, or .npz color is in range [0, 255])...")
        if ext == 'ply':
            pcd_ply_reader = PointcloudPlyFormat()
            point_cloud = pcd_ply_reader.read(args.file_name, device=device, path_manager=PyPathManager)
        elif ext == 'txt' or ext == 'xyz':  # Assume .txt or .xyz is a text file containing (N, 6) array with [XYZ RGB]
            np_pcd = np.loadtxt(args.file_name)
            verts = torch.tensor(np_pcd[:, :3], device=device, dtype=torch.float)
            rgb = torch.tensor(np_pcd[:, 3:], device=device, dtype=torch.float) / 255.
            point_cloud = Pointclouds(points=[verts], features=[rgb])
        elif ext == 'npz':  # Assume .npz is a (N, 6) array with [XYZ RGB]
            np_pcd = np.load(args.file_name)
            verts = torch.tensor(np_pcd[:, :3], device=device, dtype=torch.float)
            rgb = torch.tensor(np_pcd[:, 3:], device=device, dtype=torch.float) / 255.
            point_cloud = Pointclouds(points=[verts], features=[rgb])
        else:
            raise NotImplementedError("Other mesh data files not supported")
        print("Done!")

        point_cloud = point_cloud.to(device)

        raster_settings = PointsRasterizationSettings(
            image_size=512, 
            radius = 0.01,
            points_per_pixel = 10
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)

        renderer = PulsarPointsRenderer(
            rasterizer=rasterizer,
            n_channels=4,
            max_num_spheres=1e7
        ).to(device)

        # Add alpha channel
        for idx, feat in enumerate(point_cloud.features_list()):
            if len(feat.shape) == 2:
                point_cloud.features_list()[idx] = torch.cat([feat, torch.ones_like(feat[:, 0:1])], dim=-1)

        model_3d = point_cloud

    # Image and pose list for saving
    img_list = []
    sil_list = []
    pose_dict = {'K': cam_intrinsics, 'R': [], 'T': []}

    # Important flags for keeping track of window
    RECORD = False

    # Initiate visualization
    window = tk.Tk()
    window.title('Current View')
    if args.data_type == 'pcd':
        render_result = renderer(model_3d, gamma=(1e-4,), bg_col=torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32, device=device))
        images = (render_result[0][..., :3].cpu().numpy() * 255).astype(np.uint8)
        alpha = render_result[0][..., 3].cpu().numpy()
        sil = (alpha != 0.)
        cameras.fov = torch.rad2deg(cameras.fov)  # This line is crucial, there is a bug in PyTorch3D for Pulsar
    else:
        render_result = renderer(model_3d, lights=lights)
        images = (render_result[0][..., :3].cpu().numpy() * 255).astype(np.uint8)
        alpha = render_result[0][..., 3].cpu().numpy()
        sil = (alpha != 0.)

    win_img = ImageTk.PhotoImage(Image.fromarray(images))
    lbl = tk.Label(window, image=win_img)
    lbl.pack()

    def vis_callback(e, move_type, step_size):
        if move_type == 'fwd_x':
            cameras.T[0, 0] += step_size
        elif move_type == 'bwd_x':
            cameras.T[0, 0] -= step_size
        elif move_type == 'fwd_y':
            cameras.T[0, 1] += step_size
        elif move_type == 'bwd_y':
            cameras.T[0, 1] -= step_size
        elif move_type == 'fwd_z':
            cameras.T[0, 2] += step_size
        elif move_type == 'bwd_z':
            cameras.T[0, 2] -= step_size
        elif move_type == 'fwd_yaw':
            prev_R = cameras.R.clone().detach()
            euler_R = matrix_to_euler_angles(cameras.R, 'ZYX')
            euler_R[0, 0] += step_size
            cameras.R = euler_angles_to_matrix(euler_R, 'ZYX')
            cameras.T[0] = cameras.R[0].T @ prev_R[0] @ cameras.T[0]
        elif move_type == 'bwd_yaw':
            prev_R = cameras.R.clone().detach()
            euler_R = matrix_to_euler_angles(cameras.R, 'ZYX')
            euler_R[0, 0] -= step_size
            cameras.R = euler_angles_to_matrix(euler_R, 'ZYX')
            cameras.T[0] = cameras.R[0].T @ prev_R[0] @ cameras.T[0]
        elif move_type == 'fwd_pitch':
            prev_R = cameras.R.clone().detach()
            euler_R = matrix_to_euler_angles(cameras.R, 'ZYX')
            euler_R[0, 2] += step_size
            cameras.R = euler_angles_to_matrix(euler_R, 'ZYX')
            cameras.T[0] = cameras.R[0].T @ prev_R[0] @ cameras.T[0]
        elif move_type == 'bwd_pitch':
            prev_R = cameras.R.clone().detach()
            euler_R = matrix_to_euler_angles(cameras.R, 'ZYX')
            euler_R[0, 2] -= step_size
            cameras.R = euler_angles_to_matrix(euler_R, 'ZYX')
            cameras.T[0] = cameras.R[0].T @ prev_R[0] @ cameras.T[0]
        elif move_type == 'fwd_roll':
            prev_R = cameras.R.clone().detach()
            euler_R = matrix_to_euler_angles(cameras.R, 'ZYX')
            euler_R[0, 1] += step_size
            cameras.R = euler_angles_to_matrix(euler_R, 'ZYX')
            cameras.T[0] = cameras.R[0].T @ prev_R[0] @ cameras.T[0]
        elif move_type == 'bwd_roll':
            prev_R = cameras.R.clone().detach()
            euler_R = matrix_to_euler_angles(cameras.R, 'ZYX')
            euler_R[0, 1] -= step_size
            cameras.R = euler_angles_to_matrix(euler_R, 'ZYX')
            cameras.T[0] = cameras.R[0].T @ prev_R[0] @ cameras.T[0]

        if args.data_type == 'pcd':
            render_result = renderer(model_3d, gamma=(1e-4,), bg_col=torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device))
            images = (render_result[0][..., :3].cpu().numpy() * 255).astype(np.uint8)
            alpha = render_result[0][..., 3].cpu().numpy()
            sil = (alpha != 0.)
            cameras.fov = torch.rad2deg(cameras.fov)  # This line is crucial, there is a bug in PyTorch3D for Pulsar
        else:
            render_result = renderer(model_3d, lights=lights)
            images = (render_result[0][..., :3].cpu().numpy() * 255).astype(np.uint8)
            alpha = render_result[0][..., 3].cpu().numpy()
            sil = (alpha != 0.)

        # Log images and poses
        if RECORD:
            img_list.append(images)
            sil_list.append(sil)
            pose_dict['R'].append(cameras.R.cpu().numpy().squeeze())
            pose_dict['T'].append(cameras.T.cpu().numpy().squeeze())

        win_img = ImageTk.PhotoImage(Image.fromarray(images))
        lbl.configure(image=win_img)
        lbl.image = win_img
        renderer.rasterizer.transform(model_3d, cameras=cameras)

    def vis_callback_wrapper(move_type, step_size):
        def _vis_callback(e):
            return vis_callback(e, move_type, step_size)
        return _vis_callback

    def show_controls():
        help_win = tk.Toplevel(window)
        help_win.title("Help")
        help_msg_1 = tk.Message(help_win, aspect=500, text="Pitch: W/S, Roll: A/D", justify=tk.CENTER)
        help_msg_2 = tk.Message(help_win, aspect=500, text="Horizontal: Left/Right, Vertical: Up/Down", justify=tk.CENTER)
        help_msg_3 = tk.Message(help_win, aspect=500, text="Forward: Scroll, Yaw: Click Left/Right", justify=tk.CENTER)
        help_msg_1.pack()
        help_msg_2.pack()
        help_msg_3.pack()
        help_close = tk.Button(help_win, text='Close', command=help_win.destroy)
        help_close.pack()
    
    def start_record():
        global RECORD
        RECORD = True
        record_lbl = tk.Label(window, text="Recording...", justify=tk.CENTER)
        record_lbl.pack()

    trans_step_size = 0.05
    rot_step_size = 0.05  # Unit is in degrees

    # Set buttons
    close_button = tk.Button(window, text='Close', command=window.destroy)
    record_button = tk.Button(window, text='Record', command=start_record)
    close_button.pack(side=tk.LEFT)  # side is for alignment
    record_button.pack(side=tk.LEFT)
    help_button = tk.Button(window, text='Help', command=show_controls)
    help_button.pack(side=tk.LEFT)

    # Callback functions for moving cameras
    window.bind("<Down>", vis_callback_wrapper('fwd_y', trans_step_size))
    window.bind("<Up>", vis_callback_wrapper('bwd_y', trans_step_size))
    window.bind("<Right>", vis_callback_wrapper('fwd_x', trans_step_size))
    window.bind("<Left>", vis_callback_wrapper('bwd_x', trans_step_size))
    window.bind("<Button-5>", vis_callback_wrapper('fwd_z', trans_step_size))
    window.bind("<Button-4>", vis_callback_wrapper('bwd_z', trans_step_size))
    window.bind("<W>", vis_callback_wrapper('bwd_pitch', rot_step_size))
    window.bind("<S>", vis_callback_wrapper('fwd_pitch', rot_step_size))
    window.bind("<A>", vis_callback_wrapper('bwd_roll', rot_step_size))
    window.bind("<D>", vis_callback_wrapper('fwd_roll', rot_step_size))
    window.bind("<Button-1>", vis_callback_wrapper('fwd_yaw', rot_step_size))
    window.bind("<Button-3>", vis_callback_wrapper('bwd_yaw', rot_step_size))
    window.bind("<w>", vis_callback_wrapper('bwd_pitch', rot_step_size))
    window.bind("<s>", vis_callback_wrapper('fwd_pitch', rot_step_size))
    window.bind("<a>", vis_callback_wrapper('bwd_roll', rot_step_size))
    window.bind("<d>", vis_callback_wrapper('fwd_roll', rot_step_size))

    # Set saving protocols upon closing
    window.mainloop()

    if RECORD:
        # When closed, save images and poses
        print("Saving images and poses...")

        # Poses
        pose_dict['R'] = np.stack(pose_dict['R'], axis=0)
        pose_dict['T'] = np.stack(pose_dict['T'], axis=0)
        np.savez(os.path.join(log_dir, 'pose.npz'), **pose_dict)

        # Images and Video
        size = (img_list[0].shape[1], img_list[0].shape[0])
        vid = cv2.VideoWriter(os.path.join(log_dir, f'video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
        sil_vid = cv2.VideoWriter(os.path.join(log_dir, f'sil_video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
        for idx, (img_frame, sil_frame) in enumerate(zip(img_list, sil_list)):
            cv2.imwrite(os.path.join(img_dir, f'image-{idx:04d}.jpg'), cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(sil_dir, f'sil-{idx:04d}.jpg'), sil_frame.astype(np.uint8) * 255)
            vid.write(cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB))
            sil_vid.write(cv2.cvtColor(sil_frame.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR))
        vid.release()
        sil_vid.release()

        print("Done!")
