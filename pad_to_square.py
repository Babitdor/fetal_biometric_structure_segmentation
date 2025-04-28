import numpy as np
from PIL import Image
from torchvision.io import read_image
import torch
import matplotlib.pyplot as plt


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


# img = read_image("data/train/001_HC.png")

# # Pad it
# padded_img = pad_image_to_square(img)

# # Print new shape
# print(padded_img.shape)

# # Show padded image
# tensor_image = padded_img.permute(1, 2, 0)
# plt.imshow(tensor_image)
# plt.axis("off")
# plt.show()
