# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import sys
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from downstreams.model import MVEstimator
from downstreams.engine import train_one_epoch

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

class DualOutput:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'a', encoding='utf-8', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def flat_map(cfg):
    # Compatibility Layer: Map hierarchical config to flat attributes
    cfg.output_dir = cfg.paths.output_dir
    cfg.log_dir = cfg.paths.log_dir
    
    cfg.batch_size = cfg.train.batch_size
    cfg.epochs = cfg.train.epochs
    cfg.accum_iter = cfg.train.accum_iter
    cfg.start_epoch = cfg.train.start_epoch
    cfg.lr = cfg.train.optimizer.lr
    cfg.min_lr = cfg.train.optimizer.min_lr
    cfg.blr = cfg.train.optimizer.blr
    cfg.weight_decay = cfg.train.optimizer.weight_decay
    cfg.warmup_epochs = cfg.train.optimizer.warmup_epochs
    
    cfg.num_workers = cfg.data.num_workers
    cfg.pin_mem = cfg.data.pin_mem
    
    cfg.device = cfg.system.device
    cfg.seed = cfg.system.seed
    cfg.distributed = cfg.system.distributed
    cfg.dist_on_itp = cfg.system.dist_on_itp
    cfg.dist_url = cfg.system.dist_url

    cfg.resume = cfg.train.resume
    return cfg

@hydra.main(config_path="config", config_name="ffrecon", version_base=None)
def main(cfg):
    OmegaConf.set_struct(cfg, False)

    cfg = flat_map(cfg)

    misc.init_distributed_mode(cfg)
    if misc.is_main_process():
        if cfg.output_dir:
            Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        import sys
        log_file = os.path.join(cfg.output_dir, "log.txt")
        sys.stdout = DualOutput(log_file)
        sys.stderr = sys.stdout

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(cfg).replace(', ', ',\n'))

    device = torch.device(cfg.device)

    # fix the seed for reproducibility
    seed = cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = instantiate(cfg.data.train_dataset, _recursive_=False)
    print(dataset_train)

    if True:  # cfg.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and cfg.log_dir is not None:
        os.makedirs(cfg.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=cfg.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = MVEstimator(cfg)

    model.to(device)

    model_without_ddp = model

    eff_batch_size = cfg.batch_size * cfg.accum_iter * misc.get_world_size()
    
    if cfg.lr is None:  # only base_lr is specified
        cfg.lr = cfg.blr * eff_batch_size / 256

    print("base lr: %.2e" % (cfg.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % cfg.lr)

    print("accumulate grad iterations: %d" % cfg.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    param_groups = model_without_ddp.parameters()
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(cfg, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {cfg.epochs} epochs")
    start_time = time.time()
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=cfg
        )
        misc.save_model(
            cfg, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=epoch, epoch_name='latest')
        if cfg.output_dir and (epoch % 10 == 0 or epoch + 1 == cfg.epochs):
            misc.save_model(
                cfg, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if cfg.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(cfg.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    main()