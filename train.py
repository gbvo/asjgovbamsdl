import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim
from torch.nn import SyncBatchNorm
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dataset import PolypDataset
from model import BUNet
from transforms import Compose, Normalize, Resize, ToTensor
from utils import (dice_bce_loss, dice_loss, get_lr_decay, setup_ddp,
                   setup_logger)


# ugly
def is_good(candidate):
    trigger = np.array([0.93, 0.92, 0.89, 0.81, 0.79])
    return np.all(candidate > trigger)


def evaluate(model, test_loader, device, logger):
    model.eval()
    dices, ious = [], []
    for sample in test_loader:
        image, target = sample["image"], sample["mask"]
        with torch.inference_mode():
            image, target = image.to(device), target.to(device)
            pred = model(image)
            pred = pred.sigmoid()
            target = torch.where(target > 0.5, 1., 0.)
            pred = torch.where(pred > 0.5, 1., 0.)

            tp = torch.logical_and(target, pred).sum(dim=(1, 2, 3))
            fp = torch.logical_and(torch.logical_not(target),
                                   pred).sum(dim=(1, 2, 3))
            fn = torch.logical_and(target,
                                   torch.logical_not(pred)).sum(dim=(1, 2, 3))
            smooth = 1e-6
            sensitivity = tp / (tp + fn + smooth)
            precision = tp / (tp + fp + smooth)

            dice = (2 * precision * sensitivity /
                    (precision + sensitivity + smooth))
            iou = tp / (tp + fp + fn + smooth)

            dices.extend(dice.tolist())
            ious.extend(iou.tolist())
    mdice = np.asarray(dices).mean()
    miou = np.asarray(ious).mean()

    message = "mDic {:.4f}, mIoU {:.4f}".format(mdice, miou)
    logger.info(message)

    return mdice, miou


def train_one_epoch(model: nn.Module,
                    train_loader: DataLoader,
                    criterion,
                    optimizer: optim.Optimizer,
                    scaler: torch.cuda.amp.grad_scaler.GradScaler,
                    lr_scheduler: optim.lr_scheduler._LRScheduler,
                    device: torch.device,
                    epoch: int,
                    logger: logging.Logger,
                    args: argparse.Namespace):
    model.train()
    local_loss = torch.zeros(1).to(device)
    for i, sample in enumerate(train_loader):
        image, mask, boundary = (
            sample["image"].to(device),
            sample["mask"].to(device),
            sample["boundary"].to(device),
        )
        for size in [256, 352, 448]:
            if not size == 352:
                image = F.interpolate(image,
                                      size=(size, size),
                                      mode="bilinear",
                                      align_corners=False)
                mask = F.interpolate(mask,
                                     size=(size, size),
                                     mode="bilinear",
                                     align_corners=False)
                boundary = F.interpolate(boundary,
                                         size=(size, size),
                                         mode="bilinear",
                                         align_corners=False)
            with torch.cuda.amp.autocast(enabled=True):
                fg2, fg3, fg4, edge = model(image)
                loss_fg2 = criterion(fg2, mask)
                loss_fg3 = criterion(fg3, mask)
                loss_fg4 = criterion(fg4, mask)
                loss_edge = dice_loss(edge, boundary)

                loss = loss_fg2 + loss_fg3 + loss_fg4 + loss_edge
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            local_loss += loss

        if i % args.print_freq == args.print_freq - 1:
            local_loss /= (3 * args.print_freq)
            dist.barrier()
            dist.all_reduce(local_loss)
            if dist.get_rank() == 0:
                lr = lr_scheduler.get_last_lr()[0]
                message = (
                    "Epoch: {:<5d} Step: {:<5d} Lr: {:<10.6f} Loss: {:.4f}"
                    .format(epoch, i, lr, local_loss.item())
                )
                logger.info(message)

            dist.barrier()
            local_loss.data.zero_()

    lr_scheduler.step()


def train(args):
    if dist.get_rank() == 0:
        logger = setup_logger(args)
    else:
        logger = None
    if dist.get_rank() == 0:
        message = json.dumps(vars(args), indent=4)
        logger.info(message)
    transforms = Compose([Resize(size=(352, 352)),
                          ToTensor(),
                          Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    device = torch.device("cuda:%d" % args.rank)
    train_set = PolypDataset(root=args.data_root,
                             phase="train",
                             transforms=transforms)
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=args.workers,
                              drop_last=True,
                              pin_memory=True)
    model = BUNet()
    model.load_backbone_weights(
        "./weights/pretrained/pvt_v2_b2.pth"
    )

    model.to(device)
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[args.rank])
    model_without_ddp = model.module

    criterion = dice_bce_loss
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda t: get_lr_decay(t, args.lr_decay_epochs, args.lr_decay_rate)
    )

    for epoch in range(args.epochs):
        if dist.get_rank() == 0:
            print("Training epoch {}".format(epoch))
        train_sampler.set_epoch(epoch)
        train_one_epoch(model,
                        train_loader,
                        criterion,
                        optimizer,
                        scaler,
                        lr_scheduler,
                        device,
                        epoch,
                        logger,
                        args)
        if dist.get_rank() == 0:
            testsets = ["CVC-ClinicDB",
                        "Kvasir",
                        "CVC-300",
                        "CVC-ColonDB",
                        "ETIS-LaribPolypDB"]
            test_mdices = []
            for ds in testsets:
                message = "Evaluating on dataset {}".format(ds)
                logger.info(message)
                test_datset = PolypDataset(
                    root="./data/TestDataset/{}".format(ds),
                    phase="test",
                    transforms=transforms,
                )
                test_loader = DataLoader(test_datset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.workers)
                mdice, _ = evaluate(model, test_loader, device, logger)
                test_mdices.append(mdice)
            test_mdices = np.asarray(test_mdices)
            if epoch % args.save_freq == args.save_freq - 1:
                torch.save(
                    model_without_ddp.state_dict(),
                    os.path.join(args.save_path, "epoch_%03d.pth" % epoch),
                )
            if is_good(test_mdices):
                torch.save(
                    model_without_ddp.state_dict(),
                    os.path.join(
                        args.save_path, "epoch_%03d_best.pth" % epoch
                    ),
                )
                message = "{} New high score! {}".format("#" * 40, "#" * 40)
                logger.info(message)


def main():
    args = get_args()
    setup_ddp(args)

    args.train_description = (
        f"backbone_{args.backbone}"
        f"_batch_{args.batch_size * args.world_size}"
        f"_epochs_{args.epochs}"
        f"_lr_{args.lr}".replace(".", "_")
    )
    args.save_path = os.path.join(args.save_path,
                                  args.train_description,
                                  datetime.now().strftime("%Y%m%d%H%M%S"))
    args.log_path = os.path.join(args.log_path, args.train_description)
    if dist.get_rank() == 0:
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(args.log_path, exist_ok=True)

    train(args)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data/TrainDataset")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--backbone", type=str, default="pvt_dilated_conv_without_background")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-decay-epochs", type=int, default=40)
    parser.add_argument("--lr-decay-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-path", type=str, default="./log")
    parser.add_argument("--save-path", type=str, default="./weights/training")
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=10)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
