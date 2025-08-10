# Copyright (c) 2023 42dot. All rights reserved.
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from dataset import construct_dataset
from network import *

from .base_model import BaseModel
from .geometry import Pose, ViewRendering
from .losses import DepthSynLoss, MultiCamLoss, SingleCamLoss

_NO_DEVICE_KEYS = ['idx', 'dataset_idx', 'sensor_name', 'filename']


class VFDepthAlgo(BaseModel):
    """
    Model class for "Self-supervised surround-view depth estimation with volumetric feature fusion"
    """
    def __init__(self, cfg, rank):
        super(VFDepthAlgo, self).__init__(cfg)
        self.rank = rank
        self.read_config(cfg)
        self.prepare_dataset(cfg, rank)
        self.models = self.prepare_model(cfg, rank)   
        self.losses = self.init_losses(cfg, rank)        
        self.view_rendering, self.pose = self.init_geometry(cfg, rank) 
        self.set_optimizer()
        
        if self.pretrain and rank == 0:
            self.load_weights()
        
    def read_config(self, cfg):    
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)
                
    def init_geometry(self, cfg, rank):
        view_rendering = ViewRendering(cfg, rank)
        pose = Pose(cfg)
        return view_rendering, pose
        
    def init_losses(self, cfg, rank):
        if self.aug_depth:
            loss_model = DepthSynLoss(cfg, rank)
        elif self.spatio_temporal or self.spatio:
            loss_model = MultiCamLoss(cfg, rank)
        else :
            loss_model = SingleCamLoss(cfg, rank)
        return loss_model
        
    def prepare_model(self, cfg, rank):
        models = {}
        models['pose_net'] = self.set_posenet(cfg)        
        models['depth_net'] = self.set_depthnet(cfg)

        # DDP training
        if self.ddp_enable == True:
            from torch.nn.parallel import DistributedDataParallel as DDP            
            process_group = dist.new_group(list(range(self.world_size)))
            # set ddp configuration
            for k, v in models.items():
                # sync batchnorm
                v = torch.nn.SyncBatchNorm.convert_sync_batchnorm(v, process_group)
                # DDP enable
                models[k] = DDP(v, device_ids=[rank], broadcast_buffers=True)
        return models

    def set_posenet(self, cfg):
        if self.pose_model =='fusion':
            return FusedPoseNet(cfg).cuda()
        else:
            return MonoPoseNet(cfg).cuda()    
        
    def set_depthnet(self, cfg):
        if self.depth_model == 'fusion':
            return FusedDepthNet(cfg).cuda()
        else:
            return MonoDepthNet(cfg).cuda()

    def prepare_dataset(self, cfg, rank):
        if rank == 0:
            print('### Preparing Datasets')
        
        if self.mode == 'train':
            self.set_train_dataloader(cfg, rank)
            if rank == 0 :
                self.set_val_dataloader(cfg)
                
        if self.mode == 'eval':
            self.set_eval_dataloader(cfg)

    def set_train_dataloader(self, cfg, rank):                 
        # jittering augmentation and image resizing for the training data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)), 
            'jittering': (0.2, 0.2, 0.2, 0.05),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # construct train dataset
        train_dataset = construct_dataset(cfg, 'train', **_augmentation)

        dataloader_opts = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        if self.ddp_enable:
            dataloader_opts['shuffle'] = False
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, 
                num_replicas = self.world_size,
                rank=rank, 
                shuffle=True
            ) 
            dataloader_opts['sampler'] = self.train_sampler

        self._dataloaders['train'] = DataLoader(train_dataset, **dataloader_opts)
        num_train_samples = len(train_dataset)    
        self.num_total_steps = num_train_samples // (self.batch_size * self.world_size) * self.num_epochs

    def set_val_dataloader(self, cfg):         
        # Image resizing for the validation data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # construct validation dataset
        val_dataset = construct_dataset(cfg, 'val', **_augmentation)

        dataloader_opts = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': 0,
            'pin_memory': True,
            'drop_last': True
        }

        self._dataloaders['val']  = DataLoader(val_dataset, **dataloader_opts)
    
    def set_eval_dataloader(self, cfg):  
        # Image resizing for the validation data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # construct validation dataset
        eval_dataset = construct_dataset(cfg, 'val', **_augmentation)

        dataloader_opts = {
            'batch_size': self.eval_batch_size,
            'shuffle': False,
            'num_workers': self.eval_num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        self._dataloaders['eval'] = DataLoader(eval_dataset, **dataloader_opts)

    def set_optimizer(self):
        parameters_to_train = []
        for v in self.models.values():
            parameters_to_train += list(v.parameters())

        self.optimizer = optim.Adam(
        parameters_to_train, 
            self.learning_rate
        )

        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            self.scheduler_step_size,
            0.1
        )
    
    def process_batch(self, inputs, rank):
        """
        Pass a minibatch through the network and generate images, depth maps, and losses.
        """
        for key, ipt in inputs.items():
            if key not in _NO_DEVICE_KEYS:
                if 'context' in key:
                    inputs[key] = [ipt[k].float().to(rank) for k in range(len(inputs[key]))]
                else:
                    inputs[key] = ipt.float().to(rank)

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.synchronize(rank)
            torch.cuda.reset_peak_memory_stats(rank)

        t0 = time.perf_counter()
        outputs = self.estimate_vfdepth(inputs)  # forward only
        if use_cuda:
            torch.cuda.synchronize(rank)
        dt_ms = (time.perf_counter() - t0) * 1e3

        if rank == 0:
            if use_cuda:
                peak_alloc_mb = torch.cuda.max_memory_allocated(rank) / (1024 ** 2)
                peak_res_mb = torch.cuda.max_memory_reserved(rank) / (1024 ** 2)
                print(
                    f"[infer] {dt_ms:.2f} ms | peak_alloc={peak_alloc_mb:.1f} MiB | peak_reserved={peak_res_mb:.1f} MiB")

                # accumulate 250-iter window
                if not hasattr(self, "_prof_win"):
                    self._prof_win = {"n": 0, "t": 0.0, "alloc": 0.0, "res": 0.0}
                w = self._prof_win
                w["n"] += 1;
                w["t"] += dt_ms;
                w["alloc"] += peak_alloc_mb;
                w["res"] += peak_res_mb
                if w["n"] == 250:
                    print(f"[infer-avg/250] {w['t'] / w['n']:.2f} ms | "
                          f"peak_alloc={w['alloc'] / w['n']:.1f} MiB | "
                          f"peak_reserved={w['res'] / w['n']:.1f} MiB")
                    self._prof_win = {"n": 0, "t": 0.0, "alloc": 0.0, "res": 0.0}
            else:
                print(f"[infer][CPU] {dt_ms:.2f} ms")
                if not hasattr(self, "_prof_win_cpu"):
                    self._prof_win_cpu = {"n": 0, "t": 0.0}
                w = self._prof_win_cpu
                w["n"] += 1;
                w["t"] += dt_ms
                if w["n"] == 250:
                    print(f"[infer-avg/250][CPU] {w['t'] / w['n']:.2f} ms")
                    self._prof_win_cpu = {"n": 0, "t": 0.0}

        losses = self.compute_losses(inputs, outputs)  # not included in timing
        return outputs, losses

        # outputs = self.estimate_vfdepth(inputs)
        #
        # peak_alloc = torch.cuda.max_memory_allocated(rank)
        # peak_reserved = torch.cuda.max_memory_reserved(rank)
        #
        # print(peak_alloc/(1024**2), peak_reserved/(1024**2))
        #
        #
        # losses = self.compute_losses(inputs, outputs)
        # return outputs, losses

    def estimate_vfdepth(self, inputs):
        """
        This function sets dataloader for validation in training.
        """          
        # pre-calculate inverse of the extrinsic matrix
        inputs['extrinsics_inv'] = torch.inverse(inputs['extrinsics'])
        
        # init dictionary 
        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}

        pose_pred = self.predict_pose(inputs)                
        depth_feats = self.predict_depth(inputs)

        for cam in range(self.num_cams):       
            outputs[('cam', cam)].update(pose_pred[('cam', cam)])              
            outputs[('cam', cam)].update(depth_feats[('cam', cam)])

        if self.syn_visualize:
            outputs['disp_vis'] = depth_feats['disp_vis']
            
        self.compute_depth_maps(inputs, outputs)
        return outputs

    def predict_pose(self, inputs):      
        """
        This function predicts poses.
        """          
        net = None
        if (self.mode != 'train') and self.ddp_enable:
            net = self.models['pose_net'].module
        else:
            net = self.models['pose_net']
        
        pose = self.pose.compute_pose(net, inputs)
        return pose

    def predict_depth(self, inputs):
        """
        This function predicts disparity maps.
        """                  
        net = None
        if (self.mode != 'train') and self.ddp_enable: 
            net = self.models['depth_net'].module
        else:
            net = self.models['depth_net']

        if self.depth_model == 'fusion':
            depth_feats = net(inputs)
        else:         
            depth_feats = {}
            for cam in range(self.num_cams):
                input_depth = inputs[('color_aug', 0, 0)][:, cam, ...]
                depth_feats[('cam', cam)] = net(input_depth)
        return depth_feats
    
    def compute_depth_maps(self, inputs, outputs):     
        """
        This function computes depth map for each viewpoint.
        """                  
        source_scale = 0
        for cam in range(self.num_cams):
            ref_K = inputs[('K', source_scale)][:, cam, ...]
            for scale in self.scales:
                disp = outputs[('cam', cam)][('disp', scale)]
                outputs[('cam', cam)][('depth', scale)] = self.to_depth(disp, ref_K)
                if self.aug_depth:
                    disp = outputs[('cam', cam)][('disp', scale, 'aug')]
                    outputs[('cam', cam)][('depth', scale, 'aug')] = self.to_depth(disp, ref_K)
    
    def to_depth(self, disp_in, K_in):        
        """
        This function transforms disparity value into depth map while multiplying the value with the focal length.
        """
        min_disp = 1/self.max_depth
        max_disp = 1/self.min_depth
        disp_range = max_disp-min_disp

        disp_in = F.interpolate(disp_in, [self.height, self.width], mode='bilinear', align_corners=False)
        disp = min_disp + disp_range * disp_in
        depth = 1/disp
        return depth * K_in[:, 0:1, 0:1].unsqueeze(2)/self.focal_length_scale
    
    def compute_losses(self, inputs, outputs):
        """
        This function computes losses.
        """          
        losses = 0
        loss_fn = defaultdict(list)
        loss_mean = defaultdict(float)

        # generate image and compute loss per cameara
        for cam in range(self.num_cams):
            self.pred_cam_imgs(inputs, outputs, cam)
            cam_loss, loss_dict = self.losses(inputs, outputs, cam)
            
            losses += cam_loss  
            for k, v in loss_dict.items():
                loss_fn[k].append(v)

        losses /= self.num_cams
        
        for k in loss_fn.keys():
            loss_mean[k] = sum(loss_fn[k]) / float(len(loss_fn[k]))

        loss_mean['total_loss'] = losses        
        return loss_mean

    def pred_cam_imgs(self, inputs, outputs, cam):
        """
        This function renders projected images using camera parameters and depth information.
        """                  
        rel_pose_dict = self.pose.compute_relative_cam_poses(inputs, outputs, cam)
        self.view_rendering(inputs, outputs, cam, rel_pose_dict)  