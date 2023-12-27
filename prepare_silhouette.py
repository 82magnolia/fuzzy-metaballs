import argparse
import os
import numpy as np
import trimesh
import pyrender
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", help="Root directory for saving images and poses")
    parser.add_argument("--mesh_file", help="Mesh file to use for creating silhouettes", default=None, type=str)
    parser.add_argument("--num_views", help="Number of views to use for generating silhouettes", default=20, type=int)
    parser.add_argument("--image_size", help="Size of image in height, width", type=list, default=[64, 64], nargs=2)
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
        mesh_tri = trimesh.load(args.mesh_file)

        # seems sane to fetch/estimate scale
        shape_scale = float(mesh_tri.vertices.std(0).mean())*3
        center = np.array(mesh_tri.vertices.mean(0))

        num_views = args.num_views
        image_size = args.image_size
        vfov_degrees = args.vfov_degrees

        # this balances covariance and mean optimization due to using Adam
        opt_shape_scale = 2.2
        shape_scale_mul = opt_shape_scale/shape_scale

        focal_length = 0.5 * image_size[0] / np.tan((np.pi / 180.0) * vfov_degrees / 2)
        cx = (image_size[1] - 1) / 2
        cy = (image_size[0] - 1) / 2

        np.random.seed(42)
        rand_quats = np.random.randn(num_views, 4)

        # Optionally perturb around current set of views
        rand_perturb_count = args.rand_perturb_count
        rand_perturb_scale = args.rand_perturb_scale
        rand_perturb = args.rand_perturb
        if rand_perturb:
            rand_quats = rand_quats[:, None, :] + np.random.randn(rand_perturb_count, 4)[None, :, :] * rand_perturb_scale
            rand_quats = rand_quats.reshape(-1, 4)

        rand_quats = rand_quats / np.linalg.norm(rand_quats, axis=1, keepdims=True)

        mesh = pyrender.Mesh.from_trimesh(mesh_tri)

        # Load mesh in PyTorch3D format
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        pt3d_mesh = load_objs_as_meshes([args.mesh_file], device=device)

        scene = pyrender.Scene()
        scene.add(mesh)

        pose_dict = {'K': None, 'R': [], 'T': []}

        # Setup camera
        camera = pyrender.IntrinsicsCamera(focal_length, focal_length, cx, cy, znear=0.1 * shape_scale, zfar=100 * shape_scale)
        pt3d_cameras = FoVPerspectiveCameras(device=device, znear=camera.znear, zfar=camera.zfar,
            fov=vfov_degrees, aspect_ratio=image_size[1] / image_size[0], degrees=True)
        cam_intrinsics = pt3d_cameras.compute_projection_matrix(pt3d_cameras.znear, pt3d_cameras.zfar, pt3d_cameras.fov,
            pt3d_cameras.aspect_ratio, pt3d_cameras.degrees).cpu().numpy()
        pose_dict['K'] = cam_intrinsics

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

        for q_idx, quat in tqdm(enumerate(rand_quats), desc="Generating views", total=len(rand_quats)):
            R = transforms3d.quaternions.quat2mat(quat)
            loc = np.array([0, 0, 3 * shape_scale]) @ R + center  # Camera center
            pt3d_rot = R
            pt3d_loc = np.array([0, 0, 3 * shape_scale]) @ pt3d_rot + center  # Camera center
            align_rot = np.eye(3)  # np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
            pt3d_loc = pt3d_loc @ align_rot
            pt3d_center = center @ align_rot
            # pt3d_loc[0] *= -1.
            # pt3d_loc[1] *= -1.
            cam_rot, cam_trans = look_at_view_transform(dist=3 * shape_scale, at=pt3d_center.reshape(1, 3),
                eye=pt3d_loc.reshape(1, 3), device=device)

            # Convert to pyrender rotations (PyTorch3D -> OpenGL = PyRender)
            pyr_R = cam_rot.squeeze().cpu().numpy() @ np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])

            pose = np.vstack([np.vstack([R, loc]).T, np.array([0, 0, 0, 1])])

            # Render in pyrender
            light = pyrender.SpotLight(color=np.ones(3), intensity=50.0, innerConeAngle=np.pi / 16.0, outerConeAngle=np.pi / 6.0)
            scene.add(light, pose=pose)

            scene.add(camera, pose=pose)

            r = pyrender.OffscreenRenderer(image_size[1], image_size[0])
            color, target_depth = r.render(scene)
            target_depth[target_depth == 0] = np.nan
            sil = (~np.isnan(target_depth))
            
            for node in list(scene.light_nodes):
                scene.remove_node(node)
                time.sleep(0.01)
            for node in list(scene.camera_nodes):
                scene.remove_node(node)
                time.sleep(0.01)

            # Render in Pytorch3D
            pt3d_cameras.R = cam_rot  # torch.from_numpy(pt3d_rot.T).unsqueeze(0).to(device)
            pt3d_cameras.T[0] = cam_trans  # torch.from_numpy(cam_trans).to(device)

            renderer.rasterizer.transform(pt3d_mesh, cameras=pt3d_cameras)
            pt3d_render_result = renderer(pt3d_mesh, lights=lights)

            images = (pt3d_render_result[0][..., :3].cpu().numpy() * 255).astype(np.uint8)
            alpha = pt3d_render_result[0][..., 3].cpu().numpy()
            pt3d_sil = (alpha != 0.)

            if render_mode == 'pyrender':
                pose_dict['T'].append(loc)
                pose_dict['R'].append(R)

                cv2.imwrite(os.path.join(image_dir, f'image-{q_idx:04d}.jpg'), cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
                cv2.imwrite(os.path.join(sil_image_dir, f'sil-{q_idx:04d}.jpg'), sil.astype(np.uint8) * 255)
            else:
                pose_dict['T'].append(cam_trans.squeeze().cpu().numpy())
                pose_dict['R'].append(cam_rot.squeeze().cpu().numpy())

                cv2.imwrite(os.path.join(image_dir, f'image-{q_idx:04d}.jpg'), cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
                cv2.imwrite(os.path.join(sil_image_dir, f'sil-{q_idx:04d}.jpg'), pt3d_sil.astype(np.uint8) * 255)

        pose_dict['R'] = np.stack(pose_dict['R'], axis=0)
        pose_dict['T'] = np.stack(pose_dict['T'], axis=0)
        np.savez(os.path.join(data_dir, 'sil_pose.npz'), **pose_dict)
