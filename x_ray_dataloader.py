import torch
from PIL import Image
import pathlib
from typing import Tuple

import torchvision.transforms


class XRayDataset(torch.utils.data.Dataset):

    def __init__(
        self, path: pathlib.Path, image_size: Tuple[int, int] = (256, 256), transform=torchvision.transforms.ToTensor()
    ):
        self.path = path
        self.transform = transform

        self.image_paths = list(path.glob("*.jpeg"))
        self.image_size = image_size

        self.dataset_length = len(self.image_paths)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index) -> torch.Tensor:
        path = self.image_paths[index % self.dataset_length]
        image = Image.open(path).convert("L").resize(self.image_size)
        image = self.transform(image)

        return image
