import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from dataset import FHCD
from unet import UNet


def pred_grid_form(data_path, model_path, device):
    model = UNet(in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image_dataset = FHCD(
        data_path,
        test=True,
        transform=transforms.Compose([transforms.Resize((252, 252))]),
    )

    print(f"Total dataset size: {len(image_dataset)}")

    test_dataloader = DataLoader(dataset=image_dataset, batch_size=1, shuffle=False)

    print(f"Total test dataset size: {len(test_dataloader)}")
    images = []
    pred_masks = []

    with torch.no_grad():
        for batch in test_dataloader:

            batch = batch.float().to(device)
            y_pred = torch.sigmoid(model(batch))
            pred_masks.extend((y_pred > 0.5).float().to(device))
            images.append(batch.to(device))

    for i in range(0, 4):
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(pred_masks[i].squeeze(), cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.show()


if __name__ == "__main__":
    # Define paths
    data_path = "data"  # Replace with the path to your test images
    model_path = "models/unet_model_epoch_10.pth"  # Replace with your saved model path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Call the function to visualize predictions
    pred_grid_form(data_path, model_path, device)
