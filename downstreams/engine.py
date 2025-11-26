import math
import sys
from typing import Iterable

import torch
import util.misc as misc
from downstreams.utils.train_utils import normalize_camera_extrinsics_and_points_batch
from downstreams.general import copy_data_to_device

def process_batch(batch):      

    # Normalize camera extrinsics and points. The function returns new tensors.
    normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths, local_points = \
        normalize_camera_extrinsics_and_points_batch(
            extrinsics=batch["extrinsics"],
            cam_points=batch["cam_points"],
            world_points=batch["world_points"],
            depths=batch["depths"],
            point_masks=batch["point_masks"],
        )

    # Replace the original values in the batch with the normalized ones.
    batch["extrinsics"] = normalized_extrinsics
    batch["cam_points"] = normalized_cam_points
    batch["world_points"] = normalized_world_points
    batch["depths"] = normalized_depths
    batch["local_points"] = local_points

    return batch

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=False):
            samples = process_batch(samples)
        samples = copy_data_to_device(samples, device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss_dict, _, _, _ = model(samples)

        loss = loss_dict['objective']
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0 and data_iter_step % 20 == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train/loss', loss_value_reduce, epoch_1000x)
            # log_writer.add_scalar('train/mse', loss_dict['mse'], epoch_1000x)
            # log_writer.add_scalar('train/conf_reg', loss_dict['conf_reg'], epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

            for i in ['local_pts_loss', 'normal_loss', 'trans_loss', 'rot_loss', 'camera_loss', 'objective']:
                log_writer.add_scalar(i, loss_dict[i], epoch_1000x)



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

