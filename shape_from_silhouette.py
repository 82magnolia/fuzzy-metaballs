import argparse
import os
import sys
import torch
import numpy as np
import trimesh
import pyrender
import transforms3d
from tqdm import tqdm
from glob import glob
import cv2
import jax
import jax.numpy as jnp
import fm_render
from jax.example_libraries import optimizers
from util import DegradeLR
from numpy import random


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
    parser.add_argument("--data_root", help="Root directory containing masks and poses")
    parser.add_argument("--mesh_file", help="Mesh file to use for creating silhouettes", default=None, type=str)
    parser.add_argument("--num_mixture", help="Number of mixtures to use for fuzzy balls", default=40, type=int)
    parser.add_argument("--save_type", help="Type of data for saving fuzzy balls", default='pcd', type=str)
    args = parser.parse_args()

    log_dir = args.log
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Load silhouettes
    sil_dir = os.path.join(args.data_root, 'sil_images')
    sil_files = sorted(glob(os.path.join(sil_dir, '*')))

    sil_list = []
    for sil_file in sil_files:
        sil = cv2.cvtColor(cv2.imread(sil_file), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
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

    # Set hyperparameters
    hyperparams = fm_render.hyperparams
    NUM_MIXTURE = args.num_mixture
    beta2 = jnp.float32(np.exp(hyperparams[0]))
    beta3 = jnp.float32(np.exp(hyperparams[1]))
    gmm_init_scale = 1
    render_jit = jax.jit(fm_render.render_func_rays)
    mesh_tri = trimesh.load(args.mesh_file)
    shape_scale = float(mesh_tri.vertices[0].std(0).mean()) * 3
    center = np.array(mesh_tri.vertices.mean(0))

    # this balances covariance and mean optimization due to using Adam
    opt_shape_scale = 2.2
    shape_scale_mul = opt_shape_scale / shape_scale

    # Initialize fuzzy balls
    rand_mean = center + np.random.multivariate_normal(mean=[0, 0, 0], cov=1e-2 * np.identity(3) * shape_scale, size=NUM_MIXTURE)
    rand_weight_log = jnp.log(np.ones(NUM_MIXTURE) / NUM_MIXTURE) + jnp.log(gmm_init_scale)
    rand_sphere_size = 30
    rand_prec = jnp.array([np.identity(3) * rand_sphere_size / shape_scale for _ in range(NUM_MIXTURE)])

    height, width = image_size
    K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
    if 'pyrender' in sil_dir:
        pixel_list = (np.array(np.meshgrid(np.arange(width), height - np.arange(height) - 1, [0]))[:, :, :, 0]).reshape((3, -1)).T
    else:
        pixel_list = (np.array(np.meshgrid(width - np.arange(width) - 1, height - np.arange(height) - 1, [0]))[:, :, :, 0]).reshape((3, -1)).T
    camera_rays = (pixel_list - K[:, 2])/np.diag(K)

    if 'pyrender' in sil_dir:
        camera_rays[:, -1] = -1
    else:
        camera_rays[:, -1] = 1
    cameras_list = []
    for tran, rot in zip(pose_dict['T'], pose_dict['R']):
        if 'pyrender' in sil_dir:
            camera_rays2 = camera_rays @ rot
            t = np.tile(tran[None], (camera_rays2.shape[0], 1))
        else:
            cam_center = - tran @ rot.T
            camera_rays2 = camera_rays @ rot.T  # PyTorch3D World-Cam: X' = X @ R + T (this is inverse, cam to world)
            t = np.tile(cam_center[None], (camera_rays2.shape[0], 1))

        rays_trans = np.stack([camera_rays2, t], 1)
        cameras_list.append(rays_trans)
    cam_center = np.concatenate([cameras_list[i][:, 1] for i in range(len(cameras_list))], axis=0)
    cam_screen = np.concatenate([cameras_list[i][:, 1] + cameras_list[i][:, 0] for i in range(len(cameras_list))], axis=0)

    # np.savetxt('test_center.txt', cam_center)
    # np.savetxt('test_screen.txt', cam_screen)
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
