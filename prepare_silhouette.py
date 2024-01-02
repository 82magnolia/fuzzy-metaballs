import argparse
import os
import numpy as np
import transforms3d
from tqdm import tqdm
import time
import cv2
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    PointLights,
    SoftPhongShader,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
)
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io.ply_io import PointcloudPlyFormat
import torch
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp


class PyPathManager:
    def __init__(self):
        pass
    def open(f, mode):
        return open(f, mode)
    def exists(f):
        return os.path.exists(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", help="Root directory for saving images and poses")
    parser.add_argument("--mesh_file", help="Mesh file to use for creating silhouettes", default=None, type=str)
    parser.add_argument("--pcd_file", help="Point cloud file to use for creating silhouettes", default=None, type=str)
    parser.add_argument("--num_views", help="Number of views to use for generating silhouettes if view_type is random", default=20, type=int)
    parser.add_argument("--num_view_per_wp", help="Number of views to use for generating silhouettes if view_type is continuous", default=30, type=int)
    parser.add_argument("--image_size", help="Size of image in height, width", type=int, default=[64, 64], nargs=2)
    parser.add_argument("--view_type", help="Type of views to make for shape from silhouettes", default="random")
    parser.add_argument("--num_waypoints", help="Number of waypoints to make if view_type is continous", default=4, type=int)
    parser.add_argument("--vfov_degrees", help="Vertical field of view in degrees", type=float, default=45)
    parser.add_argument("--rand_perturb", help="If True, randomly perturb pose around each uniformly sampled pose", action='store_true')
    parser.add_argument("--rand_perturb_count", help="Number of random perturbations to make per pose", default=20, type=int)
    parser.add_argument("--rand_perturb_scale", help="Scale of random perturbations to make", default=0.1, type=float)
    parser.add_argument("--add_symmetry", help="If True, additionally render the approximate point symmetry silhouette", action='store_true')
    parser.add_argument("--seed", help="Random seed to use", default=42)
    parser.add_argument("--dist_variation", help="If True, additionally vary the distance of each camera to the object center", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    data_dir = args.data_root
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    image_dir = os.path.join(data_dir, 'images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    sil_image_dir = os.path.join(data_dir, 'sil_images')
    if not os.path.exists(sil_image_dir):
        os.makedirs(sil_image_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.mesh_file is not None:  # Generate images from mesh
        # Load mesh in PyTorch3D format
        pt3d_mesh = load_objs_as_meshes([args.mesh_file], device=device)
        verts_arr = pt3d_mesh.verts_packed().cpu().numpy()
    else:  # Generate images from point cloud
        pcd_ply_reader = PointcloudPlyFormat()
        pt3d_pcd = pcd_ply_reader.read(args.pcd_file, device=device, path_manager=PyPathManager)
        verts_arr = pt3d_pcd.points_packed().cpu().numpy()

        # Add alpha channel
        for idx, feat in enumerate(pt3d_pcd.features_list()):
            if len(feat.shape) == 2:
                pt3d_pcd.features_list()[idx] = torch.cat([feat, torch.ones_like(feat[:, 0:1])], dim=-1)

    # seems sane to fetch/estimate scale
    shape_scale = float(verts_arr.std(0).mean())*3
    center = np.array(verts_arr.mean(0))

    # Prepare camera trajectories
    image_size = args.image_size
    vfov_degrees = args.vfov_degrees

    focal_length = 0.5 * image_size[0] / np.tan((np.pi / 180.0) * vfov_degrees / 2)
    cx = (image_size[1] - 1) / 2
    cy = (image_size[0] - 1) / 2

    if args.view_type == 'continuous':
        num_waypoints = args.num_waypoints
        num_view_per_wp = args.num_view_per_wp
        waypoint_quats = np.random.randn(num_waypoints, 4)
        waypoint_sp_quats = Rotation.from_quat(waypoint_quats)
        view_quats = []
        for idx in range(len(waypoint_sp_quats)):
            if idx != len(waypoint_sp_quats) - 1:
                wps = waypoint_sp_quats[idx: idx + 2]
            else:
                wps = Rotation.concatenate([waypoint_sp_quats[-1], waypoint_sp_quats[0]])
            key_times = np.array([0, 1])
            slerp_op = Slerp(key_times, wps)
            times = np.linspace(0, 1, num_view_per_wp)
            slerp_quats = slerp_op(times)
            view_quats.append(slerp_quats.as_quat()[:-1])
        view_quats = np.concatenate(view_quats, axis=0)
    else:
        num_views = args.num_views
        view_quats = np.random.randn(num_views, 4)

    # Optionally perturb around current set of views
    rand_perturb_count = args.rand_perturb_count
    rand_perturb_scale = args.rand_perturb_scale
    rand_perturb = args.rand_perturb
    if rand_perturb:
        view_quats = view_quats[:, None, :] + np.random.randn(rand_perturb_count, 4)[None, :, :] * rand_perturb_scale
        view_quats = view_quats.reshape(-1, 4)

    view_quats = view_quats / np.linalg.norm(view_quats, axis=1, keepdims=True)

    if args.dist_variation:  # Additionally apply variations on camera distance
        dist_mult_factor = 3.
        if args.view_type == 'continuous':
            num_waypoints = args.num_waypoints
            num_view_per_wp = args.num_view_per_wp
            waypoint_dists = np.random.randn(num_waypoints) / dist_mult_factor + 3 * shape_scale
            waypoint_dists = np.clip(waypoint_dists, a_min=0.0, a_max=None)
            view_dists = []
            for idx in range(len(waypoint_dists)):
                if idx != len(waypoint_dists) - 1:
                    wps = waypoint_dists[idx: idx + 2]
                else:
                    wps = np.stack([waypoint_dists[-1], waypoint_dists[0]])
                times = np.linspace(0, 1, num_view_per_wp)
                key_times = np.array([0, 1])
                lerp_dists = np.interp(times, key_times, wps)
                view_dists.append(lerp_dists[:-1])
            view_dists = np.concatenate(view_dists, axis=0)
        else:
            num_views = args.num_views
            view_dists = np.random.randn(num_views) / dist_mult_factor + 3 * shape_scale
            view_dists = np.clip(view_dists, a_min=0.0)
    else:  # Fixed view distances
        view_dists = np.ones(view_quats.shape[0]) * 3 * shape_scale

    # Dictionary for saving poses
    pose_dict = {'K': None, 'R': [], 'T': []}

    # Setup camera
    pt3d_cameras = FoVPerspectiveCameras(device=device, znear=0.1 * shape_scale, zfar=100 * shape_scale,
        fov=vfov_degrees, aspect_ratio=image_size[1] / image_size[0], degrees=True)
    cam_intrinsics = pt3d_cameras.compute_projection_matrix(pt3d_cameras.znear, pt3d_cameras.zfar, pt3d_cameras.fov,
        pt3d_cameras.aspect_ratio, pt3d_cameras.degrees).cpu().numpy()
    pose_dict['K'] = cam_intrinsics

    # Extract screen space intrinsics for use in DROID-SLAM coordinate frame
    screen_cam_intrinsics = np.zeros_like(cam_intrinsics[0][:3, :3])

    img_h, img_w = image_size
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
    with open(os.path.join(data_dir, 'droid_slam_intrinsics.txt'), 'w') as f:
        f.write(f"{screen_cam_intrinsics[0, 0]} {screen_cam_intrinsics[1, 1]} {screen_cam_intrinsics[0, 2]} {screen_cam_intrinsics[1, 2]}")

    if args.mesh_file is not None:
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=pt3d_cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=pt3d_cameras,
                lights=lights
            )
        )
    else:
        raster_settings = PointsRasterizationSettings(
            image_size=(img_h, img_w),
            radius = 0.0001,
            points_per_pixel = 10
        )
        rasterizer = PointsRasterizer(cameras=pt3d_cameras, raster_settings=raster_settings)

        renderer = PulsarPointsRenderer(
            rasterizer=rasterizer,
            n_channels=4,
            max_num_spheres=1e7
        ).to(device)

    if args.add_symmetry:
        sym_pose_dict = {'R': [], 'T': []}

    for p_idx, (quat, dist) in tqdm(enumerate(zip(view_quats, view_dists)), desc="Generating views", total=len(view_quats)):
        R = transforms3d.quaternions.quat2mat(quat)
        loc = np.array([0, 0, dist]) @ R + center  # Camera center

        # Look-at rotations and translations (object-centric views)
        cam_rot, cam_trans = look_at_view_transform(dist=dist, at=center.reshape(1, 3),
            eye=loc.reshape(1, 3), device=device)

        pose = np.vstack([np.vstack([R, loc]).T, np.array([0, 0, 0, 1])])

        # Render in Pytorch3D
        pt3d_cameras.R = cam_rot
        pt3d_cameras.T[0] = cam_trans

        if args.mesh_file is not None:
            pt3d_render_result = renderer(pt3d_mesh, lights=lights)
            renderer.rasterizer.transform(pt3d_mesh, cameras=pt3d_cameras)
        else:
            renderer.rasterizer.transform(pt3d_pcd, cameras=pt3d_cameras)
            pt3d_render_result = renderer(pt3d_pcd, gamma=(1e-4,), bg_col=torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32, device=device))
            pt3d_cameras.fov = torch.rad2deg(pt3d_cameras.fov)  # This line is crucial, there is a bug in PyTorch3D for Pulsar

        images = (pt3d_render_result[0][..., :3].cpu().numpy() * 255).astype(np.uint8)
        alpha = pt3d_render_result[0][..., 3].cpu().numpy()
        pt3d_sil = (alpha != 0.)

        pose_dict['T'].append(cam_trans.squeeze().cpu().numpy())
        pose_dict['R'].append(cam_rot.squeeze().cpu().numpy())

        cv2.imwrite(os.path.join(image_dir, f'image-{p_idx:04d}.jpg'), cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(sil_image_dir, f'sil-{p_idx:04d}.jpg'), pt3d_sil.astype(np.uint8) * 255)

        if args.add_symmetry:  # Add 180-deg rotation on z-axis (opposite facing camera)
            Rz = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
            sym_trans = -cam_trans.squeeze().cpu().numpy() @ Rz \
                - 2 * center @ cam_rot.squeeze().cpu().numpy() @ Rz
            sym_pose_dict['T'].append(sym_trans)
            sym_pose_dict['R'].append(cam_rot.squeeze().cpu().numpy() @ Rz)

            sym_images = np.flip(images, axis=1)
            sym_sil = np.flip(pt3d_sil, axis=1)

            cv2.imwrite(os.path.join(image_dir, f'image-{len(view_quats) + p_idx:04d}.jpg'), cv2.cvtColor(sym_images, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(sil_image_dir, f'sil-{len(view_quats) + p_idx:04d}.jpg'), sym_sil.astype(np.uint8) * 255)

    if args.add_symmetry:
        pose_dict['R'] = np.stack(pose_dict['R'] + sym_pose_dict['R'], axis=0)
        pose_dict['T'] = np.stack(pose_dict['T'] + sym_pose_dict['T'], axis=0)
    else:
        pose_dict['R'] = np.stack(pose_dict['R'], axis=0)
        pose_dict['T'] = np.stack(pose_dict['T'], axis=0)
    np.savez(os.path.join(data_dir, 'sil_pose.npz'), **pose_dict)
