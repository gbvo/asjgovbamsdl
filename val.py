import os

import cv2
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import PolypDataset
from model import BUNet
from transforms import Compose, Normalize, Resize, ToTensor


def evaluate(model, ds, test_loader, device, save_path):
    model.eval()
    for sample in test_loader:
        image, image_name, image_size = (
            sample["image"], sample["image_name"], sample["image_size"]
        )
        image_name = image_name[0]
        image_size = eval(image_size[0])

        with torch.inference_mode():
            image = image.to(device)
            pred = model(image)
            pred = pred.sigmoid()
            pred = F.interpolate(
                pred,
                size=(image_size[1], image_size[0]),
                mode="bilinear",
                align_corners=False
            )
            pred = torch.where(pred > 0.5, 1., 0.)
            pred = pred.cpu().numpy().squeeze(0).squeeze(0)
            cv2.imwrite(
                os.path.join(save_path, ds, "{}.png".format(image_name[:-4])),
                pred * 255,
            )


def main():
    save_path = "./results/pvt_dilated_conv"
    model = BUNet()
    device = torch.device("cuda:0")
    model.to(device)
    model.load_state_dict(
        torch.load(
            ("./weights/training/"
             "backbone_pvt_dilated_conv_batch_50_epochs_100_lr_0_0001/"
             "20221121005535/epoch_028_best.pth")
        )
    )
    transforms = Compose([
        Resize(size=(352, 352)),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])
    datasets = [
        "CVC-ClinicDB",
        "Kvasir",
        "CVC-300",
        "CVC-ColonDB",
        "ETIS-LaribPolypDB"
    ]
    for ds in datasets:
        print("Testing on %s ..." % ds)
        if not os.path.exists(os.path.join(save_path, ds)):
            os.makedirs(os.path.join(save_path, ds))
        test_set = PolypDataset(
            root="./data/TestDataset/{}".format(ds),
            phase="test",
            transforms=transforms,
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=1,
            shuffle=False,
        )
        evaluate(model, ds, test_loader, device, save_path)


if __name__ == "__main__":
    main()
