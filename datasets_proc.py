import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.datasets import MNIST
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms import ToTensor


def show_images_dataloader(out: str, data: DataLoader, cols: int = 4):
    imgs = next(iter(data))
    """ Plots some samples from the dataset """
    fig = plt.figure(figsize=(15, 15))
    for i, img in enumerate(imgs):
        plt.subplot(int(len(imgs) / cols) + 1, cols, i + 1)
        plt.imshow(np.array((img.permute(1, 2, 0) + 1) / 2 * 255, dtype=int))
        plt.axis("off")
    plt.tight_layout()
    fig.savefig(out)
    plt.close("all")


def show_images_batch(out: str, data: torch.Tensor, cols: int = 4):
    """Plots some samples from the dataset"""
    fig = plt.figure(figsize=(15, 15))
    for i in range(data.shape[0]):
        plt.subplot(int(data.shape[0] / cols) + 1, cols, i + 1)
        img = np.array(data[i].permute(1, 2, 0), dtype=int)
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.axis("off")
    plt.tight_layout()
    if out is not None:
        fig.savefig(out)


class SquarePad:
    def __call__(self, image):
        _, w, h = image.size()
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")


class TransformNoise:
    def __call__(self, image):
        t = image / 255
        t = t + 1
        t = torch.log(t)
        return t


def data_transform(img_size=224, rgb=False):
    if not rgb:
        transform = [
            SquarePad(),
            transforms.Grayscale(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            TransformNoise(),
        ]
    else:
        transform = [
            SquarePad(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            TransformNoise(),
        ]

    transform = transforms.Compose(transform)
    return transform


class UCMercedLandUseDataset(Dataset):
    def __init__(self, img_dir, transform, amount=None):
        self.classes = [
            "agricultural",
            "airplane",
            "baseballdiamond",
            "beach",
            "buildings",
            "chaparral",
            "denseresidential",
            "forest",
            "freeway",
            "golfcourse",
            "harbor",
            "intersection",
            "mediumresidential",
            "mobilehomepark",
            "overpass",
            "parkinglot",
            "river",
            "runway",
            "sparseresidential",
            "storagetanks",
            "tenniscourt",
        ]

        self.img_dir = img_dir
        self.img_paths = []
        self.transform = transform
        if amount is not None:
            match amount:
                case "train":
                    data_size = 90
                case "test":
                    data_size = 10
                case _:
                    data_size = None
        if data_size is None:
            self.paths = []
            for c in self.classes:
                curr_dir = os.path.join(self.img_dir, c)
                tmp_paths = [
                    os.path.join(c, x)
                    for x in os.listdir(curr_dir)
                    if x.endswith(".tif")
                ]
                self.paths += tmp_paths
        else:
            self.paths = []
            if amount == "train":
                for c in self.classes:
                    for i in range(0, data_size):
                        s = os.path.join(c, c + str(i).zfill(2) + ".tif")
                        self.paths.append(s)
            else:
                for c in self.classes:
                    for i in range(90, 90 + data_size):
                        s = os.path.join(c, c + str(i).zfill(2) + ".tif")
                        self.paths.append(s)
        for f in self.paths:
            if f.endswith("tif"):
                self.img_paths.append(os.path.join(self.img_dir, f))
        self.len = len(self.paths)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx >= 0
        assert idx < self.len
        image = Image.open(self.img_paths[idx])
        image = ToTensor()(image)
        image *= 255  # ToTensor scale to [0,1], breaking the transform func
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


class FolderDataset(Dataset):
    # simply load all images from a folder
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.img_paths = []
        self.transform = transform
        self.paths = os.listdir(self.img_dir)
        keys = []
        for f in self.paths:
            if f.endswith("png") or f.endswith("jpg") or f.endswith("jpeg"):
                self.img_paths.append(os.path.join(self.img_dir, f))
                keys.append(int(f.split(".")[0]))
        indices = np.argsort(keys)
        self.img_paths = [self.img_paths[idx] for idx in indices]
        self.len = len(self.img_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx >= 0
        assert idx < self.len
        image = read_image(self.img_paths[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


class CelebADataset(Dataset):
    def __init__(self, img_dir, transform, amount=None):
        self.img_dir = img_dir
        self.img_paths = []
        self.transform = transform
        if amount is not None:
            match amount:
                case "tiny":
                    data_size = 2000
                case "small":
                    data_size = 20000
                case "large":
                    data_size = 100000
                case "test":
                    data_size = 2096
                case _:
                    data_size = None
        if data_size is None:
            self.paths = os.listdir(self.img_dir)
        else:
            self.paths = []
            if amount != "test":
                for i in range(1, data_size + 1):
                    s = os.path.join(self.img_dir, str(i).zfill(6) + ".jpg")
                    self.paths.append(s)
            else:
                for i in range(100000, data_size + 100000):
                    s = os.path.join(self.img_dir, str(i).zfill(6) + ".jpg")
                    self.paths.append(s)
        for f in self.paths:
            if f.endswith("jpg"):
                self.img_paths.append(os.path.join(self.img_dir, f))

        self.len = len(self.img_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx >= 0
        assert idx < self.len
        image = read_image(self.img_paths[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, 0
