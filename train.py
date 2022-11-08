import argparse
import logging
import os

import numpy as np
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import PolypDataset
from model import BUNet
from datetime import datetime

from transforms import Compose, Resize, ToTensor, Normalize


def get_lr_decay(epoch, max_epoch):
    return (1 - epoch / max_epoch) ** 0.9


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
    print(message)
    logger.info(message)

    return mdice, miou


def train_one_epoch(model: nn.Module,
                    train_loader: DataLoader,
                    criterion,
                    optimizer: optim.Optimizer,
                    lr_scheduler: optim.lr_scheduler._LRScheduler,
                    device: torch.device,
                    epoch: int,
                    logger: logging.Logger,
                    args: argparse.Namespace):
    model.train()
    local_loss = 0.
    for i, sample in enumerate(train_loader):
        image, mask, imask, boundary = (
            sample["image"].to(device),
            sample["mask"].to(device),
            sample["imask"].to(device),
            sample["boundary"].to(device),
        )
        for size in [256, 352, 448]:
            image = F.interpolate(image,
                                  size=(size, size),
                                  mode="bilinear",
                                  align_corners=False)
            mask = F.interpolate(mask,
                                 size=(size, size),
                                 mode="bilinear",
                                 align_corners=False)
            imask = F.interpolate(imask,
                                  size=(size, size),
                                  mode="bilinear",
                                  align_corners=False)
            boundary = F.interpolate(boundary,
                                     size=(size, size),
                                     mode="bilinear",
                                     align_corners=False)

            fg2, fg3, fg4, bg2, bg3, bg4, edge = model(image)
            loss_fg2 = criterion(fg2, mask)
            loss_fg3 = criterion(fg3, mask)
            loss_fg4 = criterion(fg4, mask)
            loss_bg2 = criterion(bg2, imask)
            loss_bg3 = criterion(bg3, imask)
            loss_bg4 = criterion(bg4, imask)
            loss_edge = dice_loss(edge, boundary)

            loss = (loss_fg2 + loss_fg3 + loss_fg4 +
                    loss_bg2 + loss_bg3 + loss_bg4 + loss_edge)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            local_loss += loss.item()

        if i % args.print_freq == args.print_freq - 1:
            local_loss = local_loss / (3 * args.print_freq)
            lr = lr_scheduler.get_last_lr()
            message = (
                "Epoch: {:<5d} Step: {:<5d} Lr: {:<10.4f} Loss: {:.4f}"
                .format(epoch, i, lr[0], local_loss)
            )
            print(message)
            logger.info(message)
            local_loss = 0.

    lr_scheduler.step()


def train(args, logger):
    transforms = Compose([Resize(size=(352, 352)),
                          ToTensor(),
                          Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    device = torch.device(args.device)
    model = BUNet()
    model.load_backbone_weights(os.path.join(args.pretrained_path))

    model.to(device)
    train_set = PolypDataset(root=args.data_root,
                             phase="train",
                             transforms=transforms)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              num_workers=args.workers,
                              shuffle=True)
    criterion = dice_bce_loss
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda t: get_lr_decay(t, args.epochs)
    )

    best_test_avg_mdice = 0.
    for epoch in range(args.epochs):
        print("Training epoch {}".format(epoch))
        train_one_epoch(model,
                        train_loader,
                        criterion,
                        optimizer,
                        lr_scheduler,
                        device,
                        epoch,
                        logger,
                        args)
        testsets = ["CVC-ClinicDB",
                    "Kvasir",
                    "CVC-300",
                    "CVC-ColonDB",
                    "ETIS-LaribPolypDB"]
        test_mdices = []
        for ds in testsets:
            message = "Evaluating on dataset {}".format(ds)
            print(message)
            logger.info(message)
            test_datset = PolypDataset(root="./data/TestDataset/{}".format(ds),
                                       phase="test",
                                       transforms=transforms)
            test_loader = DataLoader(test_datset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.workers)
            mdice, _ = evaluate(model, test_loader, device, logger)
            test_mdices.append(mdice)
        test_avg_mdice = np.asarray(test_mdices).mean()
        if epoch % args.save_freq == args.save_freq - 1:
            torch.save(
                model.state_dict(),
                os.path.join(args.save_path, "epoch_%03d.pth" % epoch),
            )
        if test_avg_mdice > best_test_avg_mdice:
            best_test_avg_mdice = test_avg_mdice
            torch.save(
                model.state_dict(),
                os.path.join(args.save_path, "epoch_%03d_best.pth" % epoch),
            )
            message = "{} New high score! {}".format("#" * 40, "#" * 40)
            print(message)
            logger.info(message)


def main():
    args = get_args()
    logger = setup_logger(args)

    train(args, logger)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data/TrainDataset")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-path", type=str, default="./log")
    parser.add_argument("--pretrained-path", type=str,
                        default="./weights/pretrained/pvt_v2_b2.pth")
    parser.add_argument("--save-path", type=str, default="./weights/training")
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=10)

    args = parser.parse_args()
    args.train_description = (f"batch_{args.batch_size}"
                              f"_epochs_{args.epochs}"
                              f"_lr_{args.lr}".replace(".", "_"))
    args.save_path = os.path.join(args.save_path,
                                  args.train_description,
                                  datetime.now().strftime("%Y%m%d%H%M%S"))
    args.log_path = os.path.join(args.log_path, args.train_description)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    return args


def setup_logger(args):
    log_file = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.log")
    logger = logging.getLogger("train")
    file_handler = logging.FileHandler(os.path.join(args.log_path, log_file))
    formatter = logging.Formatter("%(asctime)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel("INFO")

    return logger


if __name__ == "__main__":
    main()
