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

# from skimage import color, io

# from habitat_sim.utils.common import d3_40_colors_rgb

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors


import torch
import torch.nn.functional as F

# from lpips import LPIPS
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
from simulator import Simulator

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

        # if self.USE_PROFILE:
        #     print(
        #         prof.key_averages(group_by_stack_n=True).table(
        #             sort_by="self_cuda_time_total", row_limit=20
        #         )
        #     )

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

        # del out, l1_loss, depth_loss, ssim_loss, camera, rgb, depth, mask
        # torch.cuda.empty_cache()

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

        with open(f"config_" + args.habitat_scene + ".yaml", "r") as f:
            self.config_file = yaml.safe_load(f)

        self.save_path = self.config_file["save_path"] + "/"

        # self.learning_rate_lst = []

        # scene parameters
        # self.aabb = torch.tensor(
        #     self.config_file["aabb"], device=self.config_file["cuda"]
        # )

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

        data_path = os.path.join(os.getcwd(), "data")
        path_to_scene_file = os.path.join(
            data_path, "versioned_data/habitat_test_scenes/apartment_1.glb"
        )

        self.sim = Simulator(
            # args.habitat_scene,
            path_to_scene_file
        )

        self.planning_step = 25

        self.running_hessian = None

        self.quad_traj = []

        self.reg_lam = 1e-6

        self.model_params = None

        self.device = "cuda"

        print("Parameters Loaded")

    def initialization(self):
        print("initialization Started")
        sampled_poses_quat = []
        sampled_poses_mat = []
        r = R.from_quat(self.global_origin[3:])
        g_pose = self.global_origin.copy()
        initial_sample = 40
        for i in range(initial_sample):
            angles = r.as_euler("zyx", degrees=True)
            angles[1] = (angles[1] + 9 * i) % 360
            pose = g_pose.copy()
            self.quad_traj.append(np.concatenate((pose[:3], angles)))
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
        ) = self.sim.sample_images_from_poses(sampled_poses_mat)

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
        # device = (
        #     self.config_file["cuda"]
        # if model_idx == 0
        # else self.config_file["cuda"]
        # )
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
                i_image=200,
                train_lr=1e-3,
                amp=False,
                fp16=False,
                results_folder="result",
                render_kwargs=render_kwargs,
            )
            # del points
            # torch.cuda.empty_cache()
        else:
            render_kwargs = {"white_bkgd": True}
            trainer = GSSTrainer(
                model=self.gaussModel,
                data=data,
                train_batch_size=1,
                train_num_steps=int(steps / 4),
                i_image=200,
                train_lr=1e-3,
                amp=False,
                fp16=False,
                results_folder="result",
                render_kwargs=render_kwargs,
            )
        # del trainset, data
        # torch.cuda.empty_cache()
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
        # plt.figure()
        # plt.imshow(rendered_image.detach().cpu().numpy())
        # plt.imshow(np.ones((256, 256)))/
        # plt.scatter(0, 0)
        # plt.savefig("testing_image")
        ones = torch.ones_like(rendered_image)
        rendered_image.backward(gradient=ones)
        # print(self.model_params[0].data)
        # print(self.model_params[0].grad)
        current_hessian = torch.cat([p.grad.reshape(-1) for p in self.model_params])
        print("nan in H: ", torch.isnan(current_hessian).any().item())
        if torch.isnan(current_hessian).any().item():
            # print("nan in H: ", torch.isnan(current_hessian).any().item())
            # plt.figure()
            # plt.imshow(rendered_image.detach().cpu().numpy())
            # plt.savefig("testing_image")
            # plt.close()
            print(
                torch.sum(torch.isnan(current_hessian)).item(),
                " / ",
                current_hessian.shape[0],
            )
        # exit()
        current_hessian = (
            torch.nan_to_num(current_hessian * current_hessian) + self.reg_lam
        )
        # del out, rendered_image, ones
        # torch.cuda.empty_cache()

        for param in self.model_params:
            if param.grad is not None:
                param.grad.zero_()

        return current_hessian

    def info_gain(self, traj):

        gain = 0
        H_sum = torch.zeros_like(self.running_hessian).cpu()
        # print(gain)
        for pose in traj:

            T = np.eye(4)
            T[:3, :3] = R.from_euler("zyx", pose[3:]).as_matrix()
            T[:3, 3] = pose[:3]
            # sample_dataset = Dataset(
            #     training=True,
            #     save_fp=self.save_path + "/train/",
            #     device=self.config_file["cuda"],
            # )

            # sample_dataset.update_data(
            #     None,
            #     None,
            #     np.array(traj),
            # )
            # info_gain_val, H_sum = self.info_gain(
            #     sample_dataset.camtoworlds
            # )
            T = torch.from_numpy(T).to(torch.float32).to(self.train_dataset.device)
            # print(T)
            cam = get_camera(T.cpu(), self.train_dataset.K.cpu()).to(
                self.train_dataset.device
            )
            # ERROR HERE
            cam = to_viewpoint_camera(cam)
            H = self.hessian_approx(cam)
            # print("nan in H: ", torch.isnan(H).any())
            pose_gain = torch.sum(H * torch.reciprocal(self.running_hessian))
            H_sum += H.cpu()
            gain += pose_gain
            # del H, pose_gain, T, cam
            # torch.cuda.empty_cache()
        average_gain = gain / len(traj)
        print(average_gain)
        # del gain
        # torch.cuda.empty_cache()
        return average_gain, H_sum

    def vis_traj(self, trajs, inds, num=0):
        points = np.array(self.quad_traj)[:, :3]
        plt.figure()
        # gaussians = self.gaussModel.get_xyz.T.detach().cpu().numpy()
        # gaussians = (gaussians.T)[np.abs(gaussians[1]) < 0.25]
        # plt.scatter(gaussians[:, 0], gaussians[:, 2], c="black", label="gaussians")

        # height = self.sim.sim.pathfinder.get_bounds()[0][1]
        # meters_per_pix = 0.01
        # map = self.sim.sim.pathfinder.get_topdown_view(meters_per_pix, height)
        # plt.imshow(map)
        colors = ["blue", "yellow", "green", "red", "purple"]
        count = 0
        for ind in inds:
            # traj_points = self.sim.convert_points_to_topdown(
            #     traj[:, :3], meters_per_pix
            # )
            traj_points = trajs[ind]
            plt.plot(
                traj_points[:, 0],
                traj_points[:, 2],
                c=colors[count],
            )
            plt.scatter(
                traj_points[:, 0],
                traj_points[:, 2],
                c=colors[count],
                label="RRT Path " + str(count + 1),
            )
            count += 1
            # plt.scatter(traj[0, 0], traj[0, 2], c="green", label="start point")
            # plt.scatter(traj[-1, 0], traj[-1, 2], c="red", label="goal point")
        plt.plot(points[:, 0], points[:, 2], c="orange", label="Traveled Path")
        plt.scatter(points[:, 0], points[:, 2], c="orange", label="Traveled Path")

        plt.legend()
        plt.savefig(self.save_path + "/traj" + str(num) + ".png")
        plt.close()

    def planning(self, training_steps_per_step):
        print("Planning Thread Started")
        last_point = self.train_dataset.camtoworlds[-1].cpu().numpy()
        self.current_pose = np.zeros(6)
        self.current_pose[:3] = last_point[:3, 3]
        self.current_pose[3:] = R.from_matrix(last_point[:3, :3]).as_euler("zyx")

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

        self.quad_traj.append(self.current_pose)

        while flag and step < self.planning_step:
            print("planning step: " + str(step))
            step += 1

            # print("sampling trajectory from: " + str(current_state))

            # xyz_state = np.copy(current_state[:3])
            # xyz_state[1] = current_state[2]
            # xyz_state[2] = current_state[1]

            # aabb = np.copy(self.aabb.cpu().numpy())
            # aabb[1] = self.aabb[2]
            # aabb[2] = self.aabb[1]
            # aabb[4] = self.aabb[5]
            # aabb[5] = self.aabb[4]

            # Sample end points using current model's Gaussian locations
            num_samples = 5
            xyzs = self.gaussModel.get_xyz  # Nx3
            sample_end_points = xyzs[
                np.random.choice(len(xyzs), num_samples, replace=False)
            ]
            sample_end_points[:, 1] = 0
            yaws = np.pi * 2 * np.random.rand(num_samples)

            bounds = []
            bounds.append(self.sim.sim.pathfinder.get_bounds()[0][[0, 2]])
            bounds.append(self.sim.sim.pathfinder.get_bounds()[1][[0, 2]])
            bounds = np.array(bounds).T

            # RRT will return one trajectory (list of points R3) for a start and end position
            # Here we will loop over ths function for all sampled points, and interpolate yaw angle orientation
            full_trajs = []
            for i in range(num_samples):
                rrt = RapidlyExploringRandomTreePlanner(
                    self.gaussModel,
                    move_distance=0.5,  # how far to move in direction of sampled point
                    k=2,  # number of Gaussians
                    z=0.0,  # height of planning [m]
                    num_points_to_check=10,  # number of points to check for collision
                    cost_collision_thresh=10.0,  # cost threshold on whether or not there is a collision based off total sampled cost across the lin
                    max_samples=1000,  # maximum number of times to try sampling a new node
                    goal_tresh=0.25,  # distance from node to goal point to be considered converged
                    bounds=bounds,
                )

                goal_point = sample_end_points[i][::2].clone().detach().cpu().numpy()
                traj_xyz = rrt.plan(self.current_pose, goal_point)  # output of RRT
                # Number of rows in the original array
                num_points_in_traj = traj_xyz.shape[0]
                zero_cols = np.zeros((num_points_in_traj, 3))
                traj_full = np.hstack((traj_xyz, zero_cols))
                current_yaw = self.current_pose[4]
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
                print(traj.shape)
                info_gain_val, H_sum = self.info_gain(
                    traj
                )  # TODO information gain function(traj) goes here, use mean info gainyaw
                H_sums.append(H_sum)
                gains.append(info_gain_val.cpu())

                # del info_gain_val, traj
                # torch.cuda.empty_cache()

                # print(
                #     "torch.cuda.memory_allocated: %fGB"
                #     % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
                # )
                # print(
                #     "torch.cuda.memory_reserved: %fGB"
                #     % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)
                # )
                # print(
                #     "torch.cuda.max_memory_reserved: %fGB"
                #     % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024)
                # )

            best_index = np.argmax(np.array(gains))
            print(copy_traj[best_index].shape)
            # if step == 5:
            best_sort_inds = (np.argsort(np.array(gains))[::-1])[:5]
            print(best_sort_inds)
            # print(gains)
            self.vis_traj(copy_traj, best_sort_inds, num=step)

            self.running_hessian += H_sums[best_index].to(self.device)

            sampled_images, sampled_depths = self.sim.sample_images_from_poses(
                copy_traj[best_index]
            )

            self.current_pose = copy_traj[best_index][-1]

            sampled_poses_mat = []
            for pose in copy_traj[best_index]:
                self.quad_traj.append(pose)
                T = np.eye(4)
                T[:3, :3] = R.from_euler("zyx", pose[3:]).as_matrix()
                T[:3, 3] = pose[:3]
                sampled_poses_mat.append(T)

            self.train_dataset.update_data(
                sampled_images, sampled_depths, sampled_poses_mat
            )

            # del sampled_images, sampled_depths, sampled_poses_mat
            # torch.cuda.empty_cache()

            print("plan finished at: " + str(self.current_pose))

            # print(
            #     "torch.cuda.memory_allocated: %fGB"
            #     % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
            # )
            # print(
            #     "torch.cuda.memory_reserved: %fGB"
            #     % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)
            # )
            # print(
            #     "torch.cuda.max_memory_reserved: %fGB"
            #     % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024)
            # )

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
        self.gauss_training(
            int(self.config_file["training_steps"] / 50), initial_train=True
        )
        # self.gaussModel.create_from_pcd(pcd=raw_points)

        self.model_params = (
            self.gaussModel._xyz,
            self.gaussModel._features_dc,
            # self.gaussModel._features_rest,
            self.gaussModel._scaling,
            self.gaussModel._rotation,
            self.gaussModel._opacity,
        )

        plan = True

        if plan:

            self.planning(int(self.config_file["training_steps"]))

            self.gauss_training(
                self.config_file["training_steps"] * 5, final_train=True
            )

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

        # self.gaussModel.save_ply(
        #     self.save_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # )

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

    # random.seed(9)
    # np.random.seed(9)
    # torch.manual_seed(9)

    mapper = ActiveGaussSplatMapper(args)
    mapper.pipeline()
