import logging
import os
from datetime import datetime

import torch
from torch import distributed as dist
from torch.nn import functional as F


def get_lr_decay(epoch, decay_epochs, decay_rate):
    return decay_rate ** (epoch // decay_epochs)


def dice_bce_loss(inputs, targets):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(targets, kernel_size=31, stride=1, padding=15) - targets
    )
    wbce = F.binary_cross_entropy_with_logits(inputs,
                                              targets,
                                              reduction='none')
    inputs = torch.sigmoid(inputs)
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    inter = ((inputs * targets) * weit).sum(dim=(2, 3))
    union = ((inputs + targets) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def dice_loss(inputs, targets):
    smooth = 1
    inputs = torch.sigmoid(inputs)
    intersection = (targets * inputs).sum(dim=(2, 3))
    dice = ((2.0 * intersection + smooth) /
            (inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth))

    return (1 - dice).mean()


def setup_ddp(args):
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl",
                            rank=args.rank,
                            world_size=args.world_size,
                            init_method="env://")


def setup_logger(args):
    log_file = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.log")
    logger = logging.getLogger("train")
    file_handler = logging.FileHandler(os.path.join(args.log_path, log_file))
    formatter = logging.Formatter("%(asctime)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel("INFO")

    return logger
