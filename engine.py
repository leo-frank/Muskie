import math
import sys
from typing import Iterable
from collections import defaultdict

import torch

import util.misc as misc


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

    for data_iter_step, data_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = {k: v[0] for k, v in data_dict.items()}

        dataset_str = []
        for k, v in data_dict.items():
            dataset_str.extend(v[1])

        samples = {k: s.to(device, non_blocking=True) for k, s in samples.items()}

        for j in range(2): # sample reuse
            data_iter_step += j / 2
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                misc.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

            samples = {k: s[:, torch.randperm(s.shape[1])] for k, s in samples.items()}

            with torch.cuda.amp.autocast():
                loss_dict = defaultdict(float)
                per_instance_loss = []
                for res, s in samples.items():
                    _loss_dict, _per_instance_loss, _, _, _ = model(s, random_aspect_ratio=True, random_num_views=args.dynamic_batch,
                                            mask_mode=args.mask_mode, mask_ratio=args.mask_ratio)
                    for k in _loss_dict:
                        loss_dict[k] = loss_dict[k] + _loss_dict[k] / len(samples)
                    per_instance_loss.append(_per_instance_loss)

            loss = loss_dict['loss']
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

            metric_logger.update(loss=loss_value,
                                mse=loss_dict['mse'].item(),
                                conf_reg=loss_dict['conf_reg'].item())

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0 and data_iter_step % 20 == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('train/loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('train/mse', loss_dict['mse'], epoch_1000x)
                log_writer.add_scalar('train/conf_reg', loss_dict['conf_reg'], epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

                per_instance_loss = torch.cat(per_instance_loss)
                for dname in set(dataset_str):
                    idx = [i for i, d in enumerate(dataset_str) if d == dname]
                    log_writer.add_scalar(f'train_detail/{dname}_loss', per_instance_loss[idx].mean(), epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
