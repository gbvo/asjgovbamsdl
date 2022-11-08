import os
from torch.utils.data import Dataset
from PIL import Image


class PolypDataset(Dataset):
    def __init__(self, root, phase, transforms):
        super().__init__()
        self.root = root
        self.phase = phase
        self.transforms = transforms
        self.imgs = [
            os.path.join(self.root, "images", img)
            for img in os.listdir(os.path.join(self.root, "images"))
        ]

        self.masks = [img.replace("images", "masks") for img in self.imgs]
        self.imasks = [img.replace("images", "inverted_masks")
                       for img in self.imgs]
        self.boundaries = [img.replace("images", "boundaries")
                           for img in self.imgs]

    def __getitem__(self, idx):
        image_path = self.imgs[idx]
        mask_path = self.masks[idx]
        imask_path = self.imasks[idx]
        boundary_path = self.boundaries[idx]
        image_name = os.path.basename(image_path)
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        imask = Image.open(imask_path).convert("L")
        boundary = Image.open(boundary_path).convert("L")
        image_size = str(image.size)
        sample = {"image": image,
                  "mask": mask,
                  "imask": imask,
                  "boundary": boundary}
        sample = self.transforms(sample)
        if self.phase == "train":
            return sample
        else:
            sample.update({"image_name": image_name, "image_size": image_size})
            return sample

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transforms import (
        Compose,
        Resize,
        ToTensor,
        Normalize,
    )

    root = "./data/TrainDataset"
    transforms = Compose([
        Resize(size=(352, 352)),
        ToTensor(),
        Normalize(
            mean=[.5, .5, .5],
            std=[.1, .1, .1],
        ),
    ])
    dataset = PolypDataset(root=root, phase="train", transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    for sample in dataloader:
        image, target = sample["image"], sample["boundary"]
        print(image.shape, target.shape)
