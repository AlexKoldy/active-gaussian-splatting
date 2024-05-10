# general
import argparse
import pathlib
import time
import datetime
import copy
import imageio
import tqdm
import pdb
import curses
import sys
import os
import random
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

# matplotlib.use("Agg")
import matplotlib
from skimage import color, io

# from habitat_sim.utils.common import d3_40_colors_rgb

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors


import torch
import torch.nn.functional as F
from lpips import LPIPS
import cv2


from gaussian_splatting.trainer import Trainer
import gaussian_splatting.utils.loss_utils as loss_utils
from gaussian_splatting.utils.data_utils import get_camera, read_all
from gaussian_splatting.utils.camera_utils import to_viewpoint_camera
from gaussian_splatting.utils.point_utils import get_point_clouds
from gaussian_splatting.gauss_model import GaussModel
from gaussian_splatting.gauss_render import GaussRenderer
from habitat_to_data import Dataset
import contextlib

# from torch.profiler import profile, ProfilerActivity
import gaussian_splatting.utils as utils


# habitat simulator
sys.path.append("simulator")
from simulator import HabitatSim

from rapidly_exploring_random_tree_planner import RapidlyExploringRandomTreePlanner

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--sem-num",
    #     type=int,
    #     default=0,
    #     help="number of semantics classes",
    # )
    parser.add_argument(
        "--habitat-scene",
        type=str,
        default="102344250",
        help="habitat scene",
    )
    parser.add_argument(
        "--habitat-config-file",
        type=str,
        default=str(
            pathlib.Path.cwd()
            / "data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json"
        ),
        help="scene_dataset_self.config_file",
    )
    return parser.parse_args()


class GSSTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = kwargs.get("data")
        self.gaussRender = GaussRenderer(**kwargs.get("render_kwargs", {}))
        self.lambda_dssim = 0.2
        self.lambda_depth = 0.0
        self.USE_GPU_PYTORCH = True
        # self.USE_PROFILE = False

    def on_train_step(self):
        ind = np.random.choice(len(self.data["camera"]))
        camera = self.data["camera"][ind]
        rgb = self.data["rgb"][ind]
        depth = self.data["depth"][ind]
        mask = self.data["alpha"][ind] > 0.5
        if self.USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)

        # if self.USE_PROFILE:
        #     prof = profile(activities=[ProfilerActivity.CUDA], with_stack=True)
        # else:
        prof = contextlib.nullcontext()

        with prof:
            out = self.gaussRender(pc=self.model, camera=camera)

        if self.USE_PROFILE:
            print(
                prof.key_averages(group_by_stack_n=True).table(
                    sort_by="self_cuda_time_total", row_limit=20
                )
            )

        l1_loss = loss_utils.l1_loss(out["render"], rgb)
        depth_loss = loss_utils.l1_loss(out["depth"][..., 0][mask], depth[mask])
        ssim_loss = 1.0 - loss_utils.ssim(out["render"], rgb)

        total_loss = (
            (1 - self.lambda_dssim) * l1_loss
            + self.lambda_dssim * ssim_loss
            + depth_loss * self.lambda_depth
        )
        # psnr = utils.img2psnr(out['render'], rgb)
        log_dict = {
            "total": total_loss,
            "l1": l1_loss,
            "ssim": ssim_loss,
            "depth": depth_loss,
        }  # , 'psnr': psnr}

        return total_loss, log_dict

    def on_evaluate_step(self, **kwargs):
        import matplotlib.pyplot as plt

        ind = np.random.choice(len(self.data["camera"]))
        camera = self.data["camera"][ind]
        if self.USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)

        rgb = self.data["rgb"][ind].detach().cpu().numpy()
        out = self.gaussRender(pc=self.model, camera=camera)
        rgb_pd = out["render"].detach().cpu().numpy()
        depth_pd = out["depth"].detach().cpu().numpy()[..., 0]
        depth = self.data["depth"][ind].detach().cpu().numpy()
        depth = np.concatenate([depth, depth_pd], axis=1)
        depth = 1 - depth / depth.max()
        depth = plt.get_cmap("jet")(depth)[..., :3]
        image = np.concatenate([rgb, rgb_pd], axis=1)
        image = np.concatenate([image, depth], axis=0)
        utils.imwrite(str(self.results_folder / f"image-{self.step}.png"), image)


class ActiveGaussSplatMapper:
    def __init__(self, args) -> None:
        print("Parameters Loading")
        # initialize radiance field, estimator, optimzer, and dataset

        with open(f"scripts/config_" + args.habitat_scene + ".yaml", "r") as f:
            self.config_file = yaml.safe_load(f)

        self.save_path = (
            self.config_file["save_path"]
            + "/"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        self.learning_rate_lst = []

        # scene parameters
        self.aabb = torch.tensor(
            self.config_file["aabb"], device=self.config_file["cuda"]
        )

        self.train_dataset = None
        self.test_dataset = None

        # self.sem_ce_ls = []

        self.sim_step = 0
        self.viz_save_path = self.save_path + "/viz/"

        self.gaussModel = GaussModel(debug=False)

        self.gaussRender = GaussRenderer()

        # # Replace lpips with dssim for similarity metric
        # self.lpips_net = LPIPS(net="vgg").to(self.config_file["cuda"])
        # self.lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1

        # self.focal = (
        #     0.5 * self.config_file["img_w"] / np.tan(self.config_file["hfov"] / 2)
        # )

        # # Unsure if this cmap stuff is necessary
        # cmap = plt.cm.tab20
        # cmaplist = [cmap(i) for i in range(cmap.N)]
        # cmap1 = plt.cm.tab20b
        # cmaplist1 = [cmap1(i) for i in range(cmap1.N)]

        # cmaplist = (
        #     cmaplist
        #     + [cmaplist1[0]]
        #     + [cmaplist1[1]]
        #     + [cmaplist1[4]]
        #     + [cmaplist1[5]]
        #     + [cmaplist1[8]]
        #     + [cmaplist1[9]]
        #     + [cmaplist1[12]]
        #     + [cmaplist1[13]]
        #     + [cmaplist1[16]]
        #     + [cmaplist1[17]]
        # )
        # self.custom_cmap = matplotlib.colors.ListedColormap(cmaplist)

        r = np.arctan(np.linspace(0.5, 319.5, 320) / 320).tolist()
        r.reverse()
        l = np.arctan(-np.linspace(0.5, 319.5, 320) / 320).tolist()
        self.align_angles = np.array(r + l)

        self.global_origin = np.array(self.config_file["global_origin"])

        self.current_pose = np.array(self.config_file["global_origin"])

        self.sim = HabitatSim(
            args.habitat_scene,
            args.habitat_config_file,
            img_w=self.config_file["img_w"],
            img_h=self.config_file["img_h"],
        )

        self.planning_step = 25

        self.running_hessian = None

        self.quad_traj = []

        self.model_params = (
            self.gaussModel._xyz,
            self.gaussModel._features_dc,
            self.gaussModel._features_rest,
            self.gaussModel._scaling,
            self.gaussModel._rotation,
            self.gaussModel._opacity,
        )

        self.reg_lam = 1e-6

        print("Parameters Loaded")

    def initialization(self):
        print("initialization Started")
        sampled_poses_quat = []
        sampled_poses_mat = []
        r = R.from_quat(self.global_origin[3:])
        g_pose = self.global_origin.copy()
        initial_sample = 39
        for i in range(initial_sample):
            angles = r.as_euler("zyx", degrees=True)
            angles[1] = (angles[1] + 9 * i) % 360
            pose = g_pose.copy()
            pose[3:] = R.from_euler("zyx", angles, degrees=True).as_quat()

            # Maybe we keep this noise addition??
            pose[:3] = pose[:3] + np.random.uniform([-0.2, -0.2, -0.2], [0.2, 0.2, 0.2])
            sampled_poses_quat.append(pose)

            T = np.eye(4)
            T[:3, :3] = R.from_quat(pose[3:]).as_matrix()
            T[:3, 3] = pose[:3]
            sampled_poses_mat.append(T)

        (
            sampled_images,
            sampled_depth_images,
            # sampled_sem_images,
        ) = self.sim.sample_images_from_poses(sampled_poses_quat)

        # Updating cost map using observed depth values
        # We can take this out
        # for i, d_img in enumerate(sampled_depth_images):
        #     d_points = d_img[int(d_img.shape[0] / 2)]
        #     R_m = sampled_poses_mat[i][:3, :3]
        #     euler = R.from_matrix(R_m).as_euler("yzx")
        #     d_angles = (self.align_angles + euler[0]) % (2 * np.pi)
        #     w_loc = sampled_poses_mat[i][:3, 3]
        #     grid_loc = np.array(
        #         (w_loc - self.aabb.cpu().numpy()[:3])
        #         // self.config_file["main_grid_size"],
        #         dtype=int,
        #     )
        #     self.cost_map, visiting_map = update_cost_map(
        #         cost_map=self.cost_map,
        #         depth=d_points,
        #         angle=d_angles,
        #         g_loc=grid_loc,
        #         w_loc=w_loc,
        #         aabb=self.aabb.cpu().numpy(),
        #         resolution=self.config_file["main_grid_size"],
        #     )
        #     self.visiting_map += visiting_map

        sampled_images = sampled_images[:, :, :, :3]

        sampled_poses_mat = np.array(sampled_poses_mat)

        self.train_dataset = Dataset(
            training=True,
            save_fp=self.save_path + "/train/",
            # num_rays=self.config_file["init_batch_size"],
            # num_models=self.config_file["n_ensembles"],
            device=self.config_file["cuda"],
        )

        self.train_dataset.update_data(
            sampled_images,
            sampled_depth_images,
            # sampled_sem_images,
            sampled_poses_mat,
        )

        # test_loc = self.config_file["test_loc"]

        # test_quat = self.config_file["test_quat"]

        # test_samples = []

        # for loc in test_loc:
        #     for quat in test_quat:
        #         test_samples.append(np.array(loc + quat))

        # test_sampled_poses_mat = []
        # for p in test_samples:
        #     T = np.eye(4)
        #     T[:3, :3] = R.from_quat(p[3:]).as_matrix()
        #     T[:3, 3] = p[:3]
        #     test_sampled_poses_mat.append(T)

        # (
        #     test_sampled_images,
        #     test_sampled_depth_images,
        #     # test_sampled_sem_images,
        # ) = self.sim.sample_images_from_poses(test_samples)

        # test_sampled_images = test_sampled_images[:, :, :, :3]

        # self.test_dataset = Dataset(
        #     training=False,
        #     save_fp=self.save_path + "/test/",
        #     # num_models=self.config_file["n_ensembles"],
        #     device=self.config_file["cuda"],
        # )

        # self.test_dataset.update_data(
        #     test_sampled_images,
        #     test_sampled_depth_images,
        #     # test_sampled_sem_images,
        #     np.array(test_sampled_poses_mat),
        # )

        print("Initialization Finished")

    def gauss_training(self, steps, final_train=False, initial_train=False):
        print("3D Gaussian Model Training Started")

        # if final_train:
        #     self.schedulers = []
        #     for i in range(self.config_file["n_ensembles"]):
        #         optimizer = self.optimizers[i]
        #         self.schedulers.append(
        #             torch.optim.lr_scheduler.MultiStepLR(
        #                 optimizer,
        #                 milestones=[int(steps * 0.3), int(steps * 0.8)],
        #                 gamma=0.1,
        #             )
        #         )

        # num_test_images = self.test_dataset.size
        # test_idx = np.arange(num_test_images)

        # self.sem_ce_ls = []

        # def occ_eval_fn(x):
        #     density = radiance_field.query_density(x)
        #     return density * self.config_file["render_step_size"]

        # losses = [[], [], []]

        # for step in tqdm.tqdm(range(steps)):
        # train and record the models in the ensemble
        # ground_truth_imgs = []
        # # rendered_imgs = [[] for _ in range(num_test_images)]
        # rendered_imgs = []

        # psnrs_lst = [[] for _ in range(num_test_images)]
        # lpips_lst = [[] for _ in range(num_test_images)]

        # ground_truth_depth = []
        # # depth_imgs = [[] for _ in range(num_test_images)]
        # depth_imgs = []
        # # mse_dep_lst = [[] for _ in range(num_test_images)]
        # mse_dep_list = []

        # ground_truth_sem = []
        # sem_imgs = []

        # training each model
        # for model_idx, (
        #     radiance_field,
        #     estimator,
        #     optimizer,
        #     scheduler,
        #     grad_scaler,
        # ) in enumerate(
        #     zip(
        #         self.radiance_fields,
        #         self.estimators,
        #         self.optimizers,
        #         self.schedulers,
        #         self.grad_scalers,
        #     )
        # ):
        device = (
            self.config_file["cuda"]
            # if model_idx == 0
            # else self.config_file["cuda"]
        )
        device = "cuda"
        # radiance_field.train()
        # estimator.train()

        trainset = self.train_dataset
        data = {}
        data["rgb"] = trainset.images / 255.0  # Check this line???
        data["depth"] = trainset.depths
        data["depth_range"] = torch.Tensor([[0, 6]] * trainset.size).to(device)
        data["alpha"] = torch.ones(trainset.depths.shape).to(device)

        data["camera"] = get_camera(trainset.camtoworlds.cpu(), trainset.K.cpu()).to(
            device
        )
        # print(data['camera'].shape)
        if initial_train:
            points = get_point_clouds(
                data["camera"],
                trainset.depths,
                torch.ones(trainset.depths.shape).to(device),
                trainset.images / 255.0,
            )
            raw_points = points.random_sample(2**12)
            self.gaussModel.create_from_pcd(pcd=raw_points)
        render_kwargs = {"white_bkgd": True}
        trainer = GSSTrainer(
            model=self.gaussModel,
            data=data,
            train_batch_size=1,
            train_num_steps=steps,
            i_image=100,
            train_lr=1e-3,
            amp=False,
            fp16=False,
            results_folder="result/train",
            render_kwargs=render_kwargs,
        )
        # trainer.on_evaluate_step()
        trainer.train()

        # ## Save checkpoit for video
        # if (step + 1) % 1000 == 0:
        #     self.render(np.array([self.current_pose]))
        #     if not os.path.exists(self.save_path + "/checkpoints/"):
        #         os.makedirs(self.save_path + "/checkpoints/")

        #     current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        #     checkpoint_path = (
        #         self.save_path
        #         + "/checkpoints/"
        #         + "model_"
        #         + str(current_time)
        #         + ".pth"
        #     )
        #     save_dict = {
        #         "occ_grid": self.estimators[0].binaries,
        #         "model": self.radiance_fields[0].state_dict(),
        #         "optimizer_state_dict": self.optimizers[0].state_dict(),
        #     }
        #     torch.save(save_dict, checkpoint_path)
        #     print("Saved checkpoints at", checkpoint_path)

        # if step == steps + 1 and (
        #     (planning_step == 0) or ((planning_step + 1) % 2 == 0) or final_train
        # ):
        #     print("start evaluation")

        #     print("loss")
        #     print(np.mean(np.array(losses), axis=1))

        #     eval_path = self.save_path + "/prediction/"
        #     if not os.path.exists(eval_path):
        #         os.makedirs(eval_path)

        #     psnr_test = np.array(psnrs_lst)[:, 0]
        #     depth_mse_test = np.array(mse_dep_lst)[:, 0]
        #     sem_ce = np.array(self.sem_ce_ls)

        #     print("Mean PSNR: " + str(np.mean(psnr_test)))
        #     print("Mean Depth MSE: " + str(np.mean(depth_mse_test)))
        #     print("Mean Semantic CE: " + str(np.mean(sem_ce)))
        #     self.errors_hist.append(
        #         [
        #             planning_step,
        #             np.mean(psnr_test),
        #             np.mean(depth_mse_test),
        #             np.mean(sem_ce),
        #         ]
        #     )

    # def render(self, traj):
    #     traj1 = np.copy(traj)
    #     traj2 = np.copy(traj)
    #     step = self.sim_step

    #     render_images = np.array(self.sim.render_tpv(traj))
    #     if not os.path.exists(self.viz_save_path):
    #         os.makedirs(self.viz_save_path)
    #     for img in render_images:
    #         cv2.imwrite(self.viz_save_path + str(self.sim_step) + ".png", img)
    #         self.sim_step += 1

    #     render_images = np.array(self.sim.render_top_tpv(traj))
    #     if not os.path.exists(self.viz_save_path):
    #         os.makedirs(self.viz_save_path)
    #     if not os.path.exists(self.viz_save_path + "top/"):
    #         os.makedirs(self.viz_save_path + "top/")
    #     for s, img in enumerate(render_images):
    #         cv2.imwrite(self.viz_save_path + "top/" + str(step + s) + ".png", img)

    #     fpv_path = self.viz_save_path + "fpv/"
    #     if not os.path.exists(fpv_path):
    #         os.makedirs(fpv_path)
    #         os.makedirs(fpv_path + "gt_rgb/")
    #         os.makedirs(fpv_path + "gt_dep/")
    #         os.makedirs(fpv_path + "gt_sem/")
    #         os.makedirs(fpv_path + "pd_rgb/")
    #         os.makedirs(fpv_path + "pd_dep/")
    #         os.makedirs(fpv_path + "pd_occ/")
    #         os.makedirs(fpv_path + "pd_sem/")

    #     (
    #         sampled_images,
    #         sampled_depth_images,
    #         sampled_sem_images,
    #     ) = self.sim.sample_images_from_poses(traj1)

    #     (
    #         rgb_predictions,
    #         depth_predictions,
    #         acc_predictions,
    #         sem_predictions,
    #     ) = Dataset.render_images_from_poses(
    #         self.radiance_fields[0],
    #         self.estimators[0],
    #         traj2,
    #         self.config_file["img_w"],
    #         self.config_file["img_h"],
    #         self.focal,
    #         self.config_file["near_plane"],
    #         self.config_file["render_step_size"],
    #         1,
    #         self.config_file["cone_angle"],
    #         self.config_file["alpha_thre"],
    #         1,
    #         self.config_file["cuda"],
    #     )

    #     for st, (rgb, dep, sem, rgb_pd, dep_pd, acc_pd, sem_pd) in enumerate(
    #         zip(
    #             sampled_images,
    #             sampled_depth_images,
    #             sampled_sem_images,
    #             rgb_predictions,
    #             depth_predictions,
    #             acc_predictions,
    #             sem_predictions,
    #         )
    #     ):
    #         current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #         cv2.imwrite(
    #             fpv_path + "gt_rgb/" + str(step + st) + ".png",
    #             cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
    #         )
    #         cv2.imwrite(
    #             fpv_path + "pd_rgb/" + str(step + st) + ".png",
    #             cv2.cvtColor(np.float32(rgb_pd * 255), cv2.COLOR_RGB2BGR),
    #         )

    #         cv2.imwrite(
    #             fpv_path + "gt_dep/" + str(step + st) + ".png",
    #             np.clip(dep * 25, 0, 255),
    #         )
    #         cv2.imwrite(
    #             fpv_path + "pd_dep/" + str(step + st) + ".png",
    #             np.clip(dep_pd * 25, 0, 255),
    #         )

    #         sem = d3_40_colors_rgb[sem.flatten()].reshape(sem.shape[0], sem.shape[1], 3)
    #         cv2.imwrite(
    #             fpv_path + "gt_sem/" + str(step + st) + ".png",
    #             cv2.cvtColor(np.float32(sem), cv2.COLOR_RGB2BGR),
    #         )
    #         sem_argmax = np.argmax(sem_pd, axis=2)
    #         sem_pd = d3_40_colors_rgb[sem_argmax.flatten()].reshape(
    #             sem_argmax.shape[0], sem_argmax.shape[1], 3
    #         )
    #         cv2.imwrite(
    #             fpv_path + "pd_sem/" + str(step + st) + ".png",
    #             cv2.cvtColor(np.float32(sem_pd), cv2.COLOR_RGB2BGR),
    #         )

    #         cv2.imwrite(
    #             fpv_path + "pd_occ/" + str(step + st) + ".png",
    #             np.clip(acc_pd * 255, 0, 255),
    #         )

    def hessian_approx(self, camera):
        out = self.gaussRender(pc=self.gaussModel, camera=camera)
        rendered_image = out["render"]
        rendered_image.backward(gradient=torch.ones_like(rendered_image))

        current_hessian = torch.cat(
            [p.grad.detach().reshape(-1) for p in self.model_params]
        )

        current_hessian = current_hessian * current_hessian + self.reg_lam

        return current_hessian

    def info_gain(self, traj):
        gain = 0
        H_sum = torch.zeros_like(self.running_hessian)
        for pose in traj:
            cam = get_camera(pose, self.train_dataset.K.cpu()).to(
                self.train_dataset.device
            )
            cam = to_viewpoint_camera(cam)
            H = self.hessian_approx(cam)
            pose_gain = torch.sum(H * torch.reciprocal(self.running_hessian))
            H_sum += H
            gain += pose_gain
        average_gain = gain / len(traj)
        return average_gain, H_sum

    def planning(self, training_steps_per_step):
        print("Planning Thread Started")

        current_state = self.global_origin

        sim_step = 0

        step = 0
        flag = True

        first_camtoworld = self.train_dataset.camtoworlds[0]
        first_camera = get_camera(
            first_camtoworld.cpu(), self.train_dataset.K.cpu()
        ).to(self.train_dataset.device)
        first_camera = to_viewpoint_camera(first_camera)
        self.running_hessian = self.hessian_approx(first_camera)

        for camtoworld in self.train_dataset.camtoworlds[1:]:
            camera = get_camera(camtoworld.cpu(), self.train_dataset.K.cpu()).to(
                self.train_dataset.device
            )
            camera = to_viewpoint_camera(camera)
            self.running_hessian += self.hessian_approx(camera)

        self.quad_traj.append(current_state)

        while flag and step < self.planning_step:
            print("planning step: " + str(step))
            step += 1

            print("sampling trajectory from: " + str(current_state))

            # xyz_state = np.copy(current_state[:3])
            # xyz_state[1] = current_state[2]
            # xyz_state[2] = current_state[1]

            aabb = np.copy(self.aabb.cpu().numpy())
            aabb[1] = self.aabb[2]
            aabb[2] = self.aabb[1]
            aabb[4] = self.aabb[5]
            aabb[5] = self.aabb[4]

            # Sample end points using current model's Gaussian locations
            num_samples = 10
            xyzs = self.gaussModel.get_xyz()  # Nx3
            sample_end_points = xyzs[
                np.random.choice(len(xyzs), num_samples, replace=False)
            ]
            sample_end_points[:, 1] = 0
            yaws = np.pi * 2 * np.random.rand(10)

            ## Replace part below with RRT
            # sampled trajectories is nested list of "shape" (N, M, 3) -> N num trajs,
            # M length of each traj (this will not be standard across trajs), 3 is xyz

            # TODO

            # N_sample_traj_pose = sample_traj(
            #     voxel_grid=np.array([vg, vg1]),
            #     current_state=xyz_state,
            #     N_traj=self.config_file["num_traj"],
            #     aabb=aabb,
            #     sim=self.sim,
            #     cost_map=self.cost_map,
            #     N_sample_disc=self.config_file["sample_disc"],
            #     voxel_grid_size=self.config_file["main_grid_size"],
            #     visiting_map=self.visiting_map,
            #     save_path=self.save_path,
            # )

            # RRT will return one trajectory (list of points R3) for a start and end position
            # Here we will loop over ths function for all sampled points, and interpolate yaw angle orientation
            N_sample_traj_pose = None  # output of RRT fcn
            full_trajs = []
            for i in range(num_samples):
                rrt = RapidlyExploringRandomTreePlanner(
                    self.gaussModel,
                    move_distance=0.25,  # how far to move in direction of sampled point
                    k=1,  # number of Gaussians
                    z=0.0,  # height of planning [m]
                    num_points_to_check=10,  # number of points to check for collision
                    cost_collision_thresh=100.0,  # cost threshold on whether or not there is a collision based off total sampled cost across the lin
                    max_samples=1000,  # maximum number of times to try sampling a new node
                    goal_tresh=0.3,  # distance from node to goal point to be considered converged
                    bounds=np.array(
                        [
                            [-3.0, 3.0],
                            [-3.0, 3.0],
                        ]
                    ),  # bounds in 2D space for sampling
                )

                traj_xyz = rrt.plan(
                    current_state, sample_end_points[i]
                )  # output of RRT
                # Number of rows in the original array
                num_points_in_traj = traj_xyz.shape[0]
                zero_cols = np.zeros((num_points_in_traj, 3))
                traj_full = np.hstack((traj_xyz, zero_cols))
                current_yaw = current_state[3:]
                goal_yaw = yaws[i]
                if goal_yaw < current_yaw:
                    goal_yaw += 2 * np.pi
                traj_yaws = np.linspace(current_yaw, goal_yaw, num=num_points_in_traj)
                for j in range(num_points_in_traj):
                    traj_full[j][4] = traj_yaws[j] % (2 * np.pi)
                full_trajs.append(traj_full)

            copy_traj = full_trajs.copy()

            gains = []
            H_sums = []
            for traj in copy_traj:
                info_gain, H_sum = info_gain(
                    traj
                )  # TODO information gain function(traj) goes here, use mean info gain
                gains.append(info_gain)
                H_sums.append(H_sum)

            best_index = np.argmax(np.array(gains))

            self.running_hessian += H_sums[best_index]

            sampled_images, sampled_depths = self.sim.render_images_from_poses(
                copy_traj[best_index]
            )

            self.current_pose = copy_traj[best_index][-1]

            self.quad_traj.append(self.current_pose)

            sampled_poses_mat = []
            for pose in copy_traj[best_index]:
                T = np.eye(4)
                T[:3, :3] = R.from_quat(pose[3:]).as_matrix()
                T[:3, 3] = pose[:3]
                sampled_poses_mat.append(T)

            self.train_dataset.update_data(
                sampled_images, sampled_depths, sampled_poses_mat
            )

            print("plan finished at: " + str(current_state))

            self.gauss_training(steps=training_steps_per_step)

            # past_unc = np.array(self.trajector_uncertainty_list[:step]).astype(float)

            # unc = np.max(np.mean(past_unc, axis=2), axis=1)
            # if step >= 5:
            #     if (
            #         unc[step - 1] > 0.05
            #         and unc[step - 2] > 0.05
            #         and unc[step - 3] > 0.05
            #         and unc[step - 4] > 0.05
            #         and unc[step - 5] > 0.05
            #     ):
            #         flag = False

    def pipeline(self):
        # Initialize training set with circular trajectory
        self.initialization()

        # Train initial model with this data
        self.gauss_training(self.config_file["training_steps"])
        # self.gaussModel.create_from_pcd(pcd=raw_points)

        self.planning(int(self.config_file["training_steps"]))

        self.gauss_training(self.config_file["training_steps"] * 5, final_train=True)

        # plt.plot(np.arange(len(self.learning_rate_lst)), self.learning_rate_lst)
        # plt.savefig(self.save_path + "/learning_rate.png")

        # plt.yscale("log")
        # plt.plot(np.arange(len(self.learning_rate_lst)), self.learning_rate_lst)
        # plt.savefig(self.save_path + "/learning_rate_log.png")

        # save radiance field, estimator, and optimzer
        print("Saving Models")
        # save_model(radiance_field, estimator, "test")

        self.train_dataset.save()
        # self.test_dataset.save()

        if not os.path.exists(self.save_path + "/checkpoints/"):
            os.makedirs(self.save_path + "/checkpoints/")

        self.gaussModel.save_ply(self.save_path)

        # self.trajector_uncertainty_list = np.array(self.trajector_uncertainty_list)
        # np.save(self.save_path + "/uncertainty.npy", self.trajector_uncertainty_list)

        # self.errors_hist = np.array(self.errors_hist)
        # np.save(self.save_path + "/errors.npy", self.errors_hist)

        # for i, (radiance_field, estimator, optimizer, scheduler) in enumerate(
        #     zip(self.radiance_fields, self.estimators, self.optimizers, self.schedulers)
        # ):
        #     checkpoint_path = (
        #         self.save_path + "/checkpoints/" + "model_" + str(i) + ".pth"
        #     )
        #     save_dict = {
        #         "occ_grid": estimator.binaries,
        #         "model": radiance_field.state_dict(),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #     }
        #     torch.save(save_dict, checkpoint_path)
        #     print("Saved checkpoints at", checkpoint_path)


if __name__ == "__main__":
    args = parse_args()

    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)

    mapper = ActiveGaussSplatMapper(args)
    mapper.pipeline()
