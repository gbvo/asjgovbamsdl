from torchvision.transforms import functional as F


class Resize:
    def __init__(self, size=(352, 352)):
        self.size = size

    def __call__(self, sample):
        image = sample["image"]
        mask = sample["mask"]
        boundary = sample["boundary"]
        sample["image"] = F.resize(image, self.size)
        sample["mask"] = F.resize(mask, self.size)
        sample["boundary"] = F.resize(boundary, self.size)
        return sample


class ToTensor:
    def __call__(self, sample):
        image = sample["image"]
        mask = sample["mask"]
        boundary = sample["boundary"]
        sample["image"] = F.to_tensor(image)
        sample["mask"] = F.to_tensor(mask)
        sample["boundary"] = F.to_tensor(boundary)
        return sample


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample["image"]
        sample["image"] = F.normalize(image, mean=self.mean, std=self.std)
        return sample


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
