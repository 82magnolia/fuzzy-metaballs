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
)
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.io import load_objs_as_meshes
import torch
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", help="Root directory for saving images and poses")
    parser.add_argument("--mesh_file", help="Mesh file to use for creating silhouettes", default=None, type=str)
    parser.add_argument("--num_views", help="Number of views to use for generating silhouettes if view_type is random", default=20, type=int)
    parser.add_argument("--num_view_per_wp", help="Number of views to use for generating silhouettes if view_type is continuous", default=30, type=int)
    parser.add_argument("--image_size", help="Size of image in height, width", type=int, default=[64, 64], nargs=2)
    parser.add_argument("--view_type", help="Type of views to make for shape from silhouettes", default="random")
    parser.add_argument("--num_waypoints", help="Number of waypoints to make if view_type is continous", default=4, type=int)
    parser.add_argument("--vfov_degrees", help="Vertical field of view in degrees", type=float, default=45)
    parser.add_argument("--rand_perturb", help="If True, randomly perturb pose around each uniformly sampled pose", action='store_true')
    parser.add_argument("--rand_perturb_count", help="Number of random perturbations to make per pose", default=20, type=int)
    parser.add_argument("--rand_perturb_scale", help="Scale of random perturbations to make", default=0.1, type=float)
    
    args = parser.parse_args()

    data_dir = args.data_root
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    image_dir = os.path.join(data_dir, 'images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    sil_image_dir = os.path.join(data_dir, 'sil_images')
    if not os.path.exists(sil_image_dir):
        os.makedirs(sil_image_dir)

    if args.mesh_file is not None:  # Generate images from mesh
        # Load mesh in PyTorch3D format
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        pt3d_mesh = load_objs_as_meshes([args.mesh_file], device=device)
        verts_arr = pt3d_mesh.verts_packed().cpu().numpy()

        # seems sane to fetch/estimate scale
        shape_scale = float(verts_arr.std(0).mean())*3
        center = np.array(verts_arr.mean(0))

        image_size = args.image_size
        vfov_degrees = args.vfov_degrees

        # this balances covariance and mean optimization due to using Adam
        opt_shape_scale = 2.2
        shape_scale_mul = opt_shape_scale/shape_scale

        focal_length = 0.5 * image_size[0] / np.tan((np.pi / 180.0) * vfov_degrees / 2)
        cx = (image_size[1] - 1) / 2
        cy = (image_size[0] - 1) / 2

        np.random.seed(42)
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

        render_mode = 'pytorch'

        for q_idx, quat in tqdm(enumerate(view_quats), desc="Generating views", total=len(view_quats)):
            R = transforms3d.quaternions.quat2mat(quat)
            loc = np.array([0, 0, 3 * shape_scale]) @ R + center  # Camera center

            # Look-at rotations and translations (object-centric views)
            cam_rot, cam_trans = look_at_view_transform(dist=3 * shape_scale, at=center.reshape(1, 3),
                eye=loc.reshape(1, 3), device=device)

            pose = np.vstack([np.vstack([R, loc]).T, np.array([0, 0, 0, 1])])

            # Render in Pytorch3D
            pt3d_cameras.R = cam_rot
            pt3d_cameras.T[0] = cam_trans

            renderer.rasterizer.transform(pt3d_mesh, cameras=pt3d_cameras)
            pt3d_render_result = renderer(pt3d_mesh, lights=lights)

            images = (pt3d_render_result[0][..., :3].cpu().numpy() * 255).astype(np.uint8)
            alpha = pt3d_render_result[0][..., 3].cpu().numpy()
            pt3d_sil = (alpha != 0.)

            pose_dict['T'].append(cam_trans.squeeze().cpu().numpy())
            pose_dict['R'].append(cam_rot.squeeze().cpu().numpy())

            cv2.imwrite(os.path.join(image_dir, f'image-{q_idx:04d}.jpg'), cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(sil_image_dir, f'sil-{q_idx:04d}.jpg'), pt3d_sil.astype(np.uint8) * 255)

        pose_dict['R'] = np.stack(pose_dict['R'], axis=0)
        pose_dict['T'] = np.stack(pose_dict['T'], axis=0)
        np.savez(os.path.join(data_dir, 'sil_pose.npz'), **pose_dict)
