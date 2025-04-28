import torch
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from loss.BCEDiceLoss import BCEDiceLoss

from unet import UNet
from dataset import FHCD

transform = transforms.Compose([transforms.Resize((252, 252))])
transform1 = transforms.Compose([transforms.Resize((252, 252))])


def dice_coefficient(pred, target, smooth=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2.0 * intersection + smooth) / (union + smooth)


if __name__ == "__main__":
    Learning_rate = 1e-4
    BATCH_SIZE = 8
    EPOCHS = 10
    DATA_PATH = "data"
    MODEL_SAVE_DIR = "models"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = FHCD(DATA_PATH, transform=transform, target_transform=transform1)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        train_dataset, [0.8, 0.2], generator=generator
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    model = UNet(in_channels=1, num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Learning_rate, weight_decay=1e-4)
    criterion = BCEDiceLoss().to(device)

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            optimizer.zero_grad()

            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)

        model.eval()
        val_running_loss = 0
        val_dice_score = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = model(img)

                dice_score = dice_coefficient(y_pred, mask)
                val_dice_score += dice_score.item()

                loss = criterion(y_pred, mask)
                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)
            avg_dice_score = val_dice_score / (idx + 1)

        print("-" * 30)
        print(f"Training Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Validation Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print(f"Validation Dice Coefficient EPOCH {epoch+1}: {avg_dice_score:.4f}")
        print("-" * 30)

        model.eval()
        with torch.no_grad():
            for idx, img_mask in enumerate(val_dataloader):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = torch.sigmoid(model(img))  # Apply sigmoid to get probabilities
                pred_mask = (y_pred > 0.5).float()  # Threshold to get binary mask

                # Visualize the first batch
                if idx == 0:

                    plt.figure(figsize=(12, 4))
                    for i in range(min(4, img.size(0))):  # Visualize up to 4 images
                        plt.subplot(3, 4, i + 1)
                        plt.imshow(img[i].cpu().squeeze(), cmap="gray")
                        plt.title("Input Image")
                        plt.axis("off")

                        plt.subplot(3, 4, i + 5)
                        plt.imshow(mask[i].cpu().squeeze(), cmap="gray")
                        plt.title("Ground Truth")
                        plt.axis("off")

                        plt.subplot(3, 4, i + 9)
                        plt.imshow(pred_mask[i].cpu().squeeze(), cmap="gray")
                        plt.title("Prediction")
                        plt.axis("off")
                    plt.show()
                    break

    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"unet_model_epoch_{EPOCHS}.pth")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
