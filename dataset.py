import os
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.io import read_image


def pad_image_to_square(img, target_size=812):

    # Get original shape
    channels, height, width = img.size()

    # Calculate total padding needed
    pad_height = max(0, target_size - height)
    pad_width = max(0, target_size - width)

    # Compute padding on each side (symmetric)
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    # Apply padding
    padded_img = torch.nn.functional.pad(
        img, (left, right, top, bottom), mode="constant", value=0
    )

    return padded_img


class FHCD(Dataset):

    def __init__(
        self,
        root_path,
        test=False,
        subset_size=None,
        transform=None,
        target_transform=None,
    ):

        super().__init__()
        self.test = test
        self.subset_size = subset_size
        self.target_transform = target_transform
        self.root_path = root_path
        self.transform = transform
        self.target_transform = target_transform

        if self.test:
            test_path = os.path.join(self.root_path, "test")
            self.images = sorted(
                [
                    os.path.join(test_path, f)
                    for f in os.listdir(test_path)
                    if f.endswith(".png")
                ]
            )
            self.masks = None

        else:
            train_path = os.path.join(self.root_path, "train")
            self.images = sorted(
                [
                    os.path.join(train_path, f)
                    for f in os.listdir(train_path)
                    if f.endswith("_HC.png") and "Annotation" not in f
                ]
            )
            self.masks = sorted(
                [
                    os.path.join(train_path, f)
                    for f in os.listdir(train_path)
                    if f.endswith("_HC_Annotation.png")
                ]
            )

            if self.subset_size is not None:
                self.images = self.images[: self.subset_size]
                self.masks = self.masks[: self.subset_size]

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((512, 512)),
                    transforms.ConvertImageDtype(torch.float32),
                ]
            )
        if self.target_transform is None:
            self.target_transform = self.transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img = read_image(self.images[index])
        img = pad_image_to_square(img)
        img = self.transform(img)

        if self.test:
            return img.float()

        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, 0, 255, -1)
        mask = torch.from_numpy(mask[np.newaxis, :, :]).float()
        mask = pad_image_to_square(mask)
        mask = self.target_transform(mask)
        mask = torch.where(mask > 128, torch.tensor([1.0]), torch.tensor([0.0]))

        return img.float(), mask.float()
