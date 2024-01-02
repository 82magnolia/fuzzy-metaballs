import argparse
import os
import numpy as np
import trimesh
from tqdm import tqdm
from glob import glob
import cv2
import jax
import jax.numpy as jnp
import fm_render
from jax.example_libraries import optimizers
from util import DegradeLR
from numpy import random
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_farthest_points
from pytorch3d.io.ply_io import PointcloudPlyFormat
import torch
from prepare_silhouette import PyPathManager


def sfs_objective(params, true_alpha):
    CLIP_ALPHA = 3e-8
    means, prec, weights_log, camera_rays, beta2, beta3 = params
    render_res = render_jit(means, prec, weights_log, camera_rays, beta2, beta3)

    est_alpha = render_res[1]
    est_alpha = jnp.clip(est_alpha, CLIP_ALPHA, 1 - CLIP_ALPHA)
    mask_loss = - ((true_alpha * jnp.log(est_alpha)) + (1-true_alpha)*jnp.log(1 - est_alpha))
    return mask_loss.mean()


def irc(x):
    return int(round(x))


if __name__ == '__main__':
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument("--log", help="Name of logging directory", default="log/")
    parser.add_argument("--data_root", help="Root directory containing masks and poses", required=True)
    parser.add_argument("--mesh_file", help="Mesh file to use for creating silhouettes", default=None, type=str)
    parser.add_argument("--pcd_file", help="Point cloud file to use for creating silhouettes", default=None, type=str)
    parser.add_argument("--num_mixture", help="Number of mixtures to use for fuzzy balls", default=40, type=int)
    parser.add_argument("--save_type", help="Type of data for saving fuzzy balls", default='pcd', type=str)
    parser.add_argument("--init_type", help="Type of initialization for 3D Gaussians", default="random", type=str)
    parser.add_argument("--slam_root", help="Root directory containing SLAM poses and reconstructions", default=None)
    parser.add_argument("--resize_rate", help="Optionally resize images for faster optimization", default=1, type=int)
    args = parser.parse_args()

    log_dir = args.log
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if args.slam_root is not None:  # Use SLAM poses
        # Load silhouettes
        with open(os.path.join(args.slam_root, 'images.txt'), 'r') as f:
            sil_files = [s.strip().replace('images', 'sil_images').replace('image-', 'sil-') for s in f.readlines()]
            sil_files = [os.path.join(args.data_root, s) for s in sil_files]

        sil_list = []
        for sil_file in sil_files:
            sil = cv2.cvtColor(cv2.imread(sil_file), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
            sil = cv2.resize(sil, (sil.shape[1] // args.resize_rate, sil.shape[0] // args.resize_rate))
            sil[sil >= 0.5] = 1.  # Binarize silhouettes
            sil[sil < 0.5] = 0.
            sil = np.fliplr(np.flipud(sil))  # Convert DROID-SLAM coordinate space to PyTorch3D coordinate space via xy-reflection
            sil_list.append(sil)

        target_sil = np.stack(sil_list, axis=0)
        droid_image_size = [args.resize_rate * sil_list[0].shape[0], args.resize_rate * sil_list[0].shape[1]]  # (H, W)
        image_size = [sil_list[0].shape[0], sil_list[0].shape[1]]  # (H, W)

        # Load poses
        ext_dir = os.path.join(args.slam_root, 'poses_mtx.npy')
        int_dir = os.path.join(args.slam_root, 'intrinsics.npy')

        ext_arr = np.load(ext_dir)
        int_arr = np.load(int_dir)

        ext_arr = np.linalg.inv(ext_arr)  # DROID-SLAM saves inverse poses
        trans_arr = ext_arr[:, 0:3, 3:].squeeze(2)
        rot_arr = np.transpose(ext_arr[:, :3, :3], (0, 2, 1))

        """
        Note the flipping operation above is equivalent to the following transformation (if we omit flipping)

        R_convert = np.array([[-1., 0, 0], [0, -1., 0], [0, 0, 1.]])
        rot_arr = rot_arr @ R_convert[None, ...]
        trans_arr = trans_arr @ R_convert
        """

        # DROID-SLAM resize factor
        droid_resize_rate = np.sqrt((384 * 512) / (droid_image_size[0] * droid_image_size[1]))
        droid_intrinsic_rate = 8.0  # DROID-SLAM divides intrinsics by 8.0
        droid_mult_factor = droid_intrinsic_rate / droid_resize_rate
        trans_arr *= droid_mult_factor  # Resize translation array

        focal_length = int_arr[0, 0] / args.resize_rate * (droid_mult_factor)
        cx = int_arr[0, 2] / args.resize_rate * (droid_mult_factor)
        cy = int_arr[0, 3] / args.resize_rate * (droid_mult_factor)

        height, width = image_size
        K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
        pixel_list = (np.array(np.meshgrid(width - np.arange(width) - 1, height - np.arange(height) - 1, [0]))[:, :, :, 0]).reshape((3, -1)).T
        camera_rays = (pixel_list - K[:, 2])/np.diag(K)
        camera_rays[:, -1] = 1
        cameras_list = []
        for tran, rot in zip(trans_arr, rot_arr):
            cam_center = - tran @ rot.T
            camera_rays2 = camera_rays @ rot.T  # PyTorch3D World-Cam: X' = X @ R + T (this is inverse, cam to world)
            t = np.tile(cam_center[None], (camera_rays2.shape[0], 1))

            rays_trans = np.stack([camera_rays2, t], 1)
            rays_trans = rays_trans / (droid_mult_factor)  # Normalization required for aligning camera_rays scale with saved translation scale

            cameras_list.append(rays_trans)
        cam_center = np.concatenate([cameras_list[i][:, 1] for i in range(len(cameras_list))], axis=0)
        cam_screen = np.concatenate([cameras_list[i][:, 1] + cameras_list[i][:, 0] for i in range(len(cameras_list))], axis=0)
    else:
        # Load silhouettes
        sil_dir = os.path.join(args.data_root, 'sil_images')
        sil_files = sorted(glob(os.path.join(sil_dir, '*')))

        sil_list = []
        for sil_file in sil_files:
            sil = cv2.cvtColor(cv2.imread(sil_file), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
            sil = cv2.resize(sil, (sil.shape[1] // args.resize_rate, sil.shape[0] // args.resize_rate))
            sil_list.append(sil)

        target_sil = np.stack(sil_list, axis=0)
        image_size = [sil_list[0].shape[0], sil_list[0].shape[1]]  # (H, W)

        # Load poses
        pose_dir = os.path.join(args.data_root, 'sil_pose.npz')
        pose_dict = np.load(pose_dir)
        ndc_intrinsics = pose_dict['K']
        focal_length = ndc_intrinsics[0][0, 0] * min(image_size[0], image_size[1]) / 2
        cx = image_size[1] / 2 - ndc_intrinsics[0][0, 2] * min(image_size[0], image_size[1]) / 2
        cy = image_size[0] / 2 - ndc_intrinsics[0][1, 2] * min(image_size[0], image_size[1]) / 2

        height, width = image_size
        K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
        pixel_list = (np.array(np.meshgrid(width - np.arange(width) - 1, height - np.arange(height) - 1, [0]))[:, :, :, 0]).reshape((3, -1)).T
        camera_rays = (pixel_list - K[:, 2])/np.diag(K)
        camera_rays[:, -1] = 1
        cameras_list = []
        for tran, rot in zip(pose_dict['T'], pose_dict['R']):
            cam_center = - tran @ rot.T
            camera_rays2 = camera_rays @ rot.T  # PyTorch3D World-Cam: X' = X @ R + T (this is inverse, cam to world)
            t = np.tile(cam_center[None], (camera_rays2.shape[0], 1))

            rays_trans = np.stack([camera_rays2, t], 1)
            cameras_list.append(rays_trans)
        cam_center = np.concatenate([cameras_list[i][:, 1] for i in range(len(cameras_list))], axis=0)
        cam_screen = np.concatenate([cameras_list[i][:, 1] + cameras_list[i][:, 0] for i in range(len(cameras_list))], axis=0)

    # Set hyperparameters
    hyperparams = fm_render.hyperparams
    NUM_MIXTURE = args.num_mixture
    beta2 = jnp.float32(np.exp(hyperparams[0]))
    beta3 = jnp.float32(np.exp(hyperparams[1]))
    gmm_init_scale = 1
    render_jit = jax.jit(fm_render.render_func_rays)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load 3D model for center & scale estimation
    if args.slam_root is not None:  # Use SLAM reconstructions to initialize
        verts_arr = np.loadtxt(os.path.join(args.slam_root, 'point_cloud.txt'))[:, :3]
    else:
        if args.mesh_file is not None:  # Generate images from mesh
            # Load mesh in PyTorch3D format
            pt3d_mesh = load_objs_as_meshes([args.mesh_file], device=device)
            verts_arr = pt3d_mesh.verts_packed().cpu().numpy()
        else:  # Generate images from point cloud
            if args.pcd_file.endswith('ply'):
                pcd_ply_reader = PointcloudPlyFormat()
                pt3d_pcd = pcd_ply_reader.read(args.pcd_file, device=device, path_manager=PyPathManager)
                verts_arr = pt3d_pcd.points_packed().cpu().numpy()
            else:  # .txt ending
                verts_arr = np.loadtxt(args.pcd_file)[:, :3]

    shape_scale = float(verts_arr.std(0).mean())*3
    center = np.array(verts_arr.mean(0))

    # this balances covariance and mean optimization due to using Adam
    opt_shape_scale = 2.2
    shape_scale_mul = opt_shape_scale / shape_scale

    # Initialize fuzzy balls
    if args.init_type == 'surface':
        rand_mean = sample_farthest_points(torch.from_numpy(verts_arr).to(device).unsqueeze(0), K=NUM_MIXTURE)[0]
        rand_mean = rand_mean.squeeze(0).cpu().numpy()
    else:
        rand_mean = center + np.random.multivariate_normal(mean=[0, 0, 0], cov=1e-2 * np.identity(3) * shape_scale, size=NUM_MIXTURE)

    rand_weight_log = jnp.log(np.ones(NUM_MIXTURE) / NUM_MIXTURE) + jnp.log(gmm_init_scale)
    rand_sphere_size = 30
    rand_prec = jnp.array([np.identity(3) * rand_sphere_size / shape_scale for _ in range(NUM_MIXTURE)])

    # Initial rendering
    alpha_results_rand = []
    alpha_results_rand_depth = []
    for camera_rays in cameras_list:
        est_depth, est_alpha, est_norm, est_w = render_jit(rand_mean, rand_prec,
            rand_weight_log, camera_rays, beta2 / shape_scale, beta3)
        alpha_results_rand.append(est_alpha.reshape(image_size))
        est_depth = np.array(est_depth)
        est_depth[est_alpha < 0.5] = np.nan
        alpha_results_rand_depth.append(est_depth.reshape(image_size))

    grad_render = jax.jit(jax.value_and_grad(sfs_objective))

    # Optimize fuzzy balls
    all_cameras = jnp.array(cameras_list).reshape((-1, 2, 3))
    all_sils = jnp.array(target_sil.ravel()).astype(jnp.float32)

    # Number of optimization steps
    Nepochs = 10
    batch_size = 800
    Niter_epoch = int(np.ceil(len(all_cameras)/batch_size))

    vecM = jnp.array([[1, 1, 1], [shape_scale_mul, shape_scale_mul, shape_scale_mul]])[None]

    outer_loop = tqdm(range(Nepochs), desc=" epoch", position=0)

    adjust_lr = DegradeLR(1e-1 , 0.5, irc(Niter_epoch*0.4), irc(Niter_epoch * 0.1), -1e-4)
    opt_init, opt_update, opt_params = optimizers.adam(adjust_lr.step_func)
    tmp = [rand_mean * shape_scale_mul, rand_prec / shape_scale_mul, rand_weight_log]
    opt_state = opt_init(tmp)

    rand_idx = np.arange(len(all_cameras))

    losses = []
    done = False

    for i in outer_loop:
        np.random.shuffle(rand_idx)
        rand_idx_jnp = jnp.array(rand_idx)
        for j in tqdm(range(Niter_epoch), desc=" iteration", position=1, leave=False):
            p = opt_params(opt_state)
            idx = jax.lax.dynamic_slice(rand_idx_jnp, [j * batch_size], [batch_size])

            val, g = grad_render([p[0], p[1], p[2], vecM * all_cameras[idx], beta2 / opt_shape_scale, beta3], all_sils[idx])   
            opt_state = opt_update(i, g[:3], opt_state)

            val = float(val)
            losses.append(val)
            outer_loop.set_description("total_loss = %.3f" % val)
            if adjust_lr.add(val):
                done = True
                break
        if done:
            break

    # Normalize results
    final_mean, final_prec, final_weight_log = opt_params(opt_state)
    final_mean /= shape_scale_mul
    final_prec *= shape_scale_mul

    if args.save_type == 'pcd':  # Save as bundle of points
        points_list = []
        colors_list = []
        num_sample = 5000

        for idx in range(len(final_mean)):
            if final_weight_log[idx] > 0:  # final_weight_log keeps valid balls
                prec = final_prec[idx]  # L^T
                points_list.append(random.multivariate_normal(final_mean[idx], np.linalg.inv(prec.T @ prec), size=num_sample))
                colors = np.stack([np.random.rand(3)] * num_sample, axis=0)
                colors_list.append(colors)
        points = np.concatenate(points_list, axis=0)
        colors = np.concatenate(colors_list, axis=0)
        pcd = np.concatenate([points, colors], axis=1)

        np.savetxt(os.path.join(log_dir, 'gaussians.txt'), pcd)
