import os
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    get_transforms_value,
)
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Hyperparameters
LEARNING_RATE = 1e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 5
NUM_WORKERS = 4
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "./brain/train/images/"
TRAIN_MASK_DIR = "./brain/train/masks/"
VAL_IMG_DIR = "./brain/val/images/"
VAL_MASK_DIR = "./brain/val/masks/"
METRICS_FILE = "training_metrics.csv"


def save_metrics_to_file(
    epoch_history,
    train_loss_history,
    val_loss_history,
    train_dice_history,
    val_dice_history,
    train_pc_history,
    val_pc_history,
):
    data = {
        "epoch": [e for e in epoch_history],
        "train_loss": [
            t.item() if isinstance(t, torch.Tensor) else t for t in train_loss_history
        ],
        "val_loss": [
            v.item() if isinstance(v, torch.Tensor) else v for v in val_loss_history
        ],
        "train_dice": [
            d.item() if isinstance(d, torch.Tensor) else d for d in train_dice_history
        ],
        "val_dice": [
            d.item() if isinstance(d, torch.Tensor) else d for d in val_dice_history
        ],
        "train_pc": [
            pc.item() if isinstance(pc, torch.Tensor) else pc for pc in train_pc_history
        ],
        "val_pc": [
            pc.item() if isinstance(pc, torch.Tensor) else pc for pc in val_pc_history
        ],
    }
    df = pd.DataFrame(data)
    df.to_csv(METRICS_FILE, index=False)


def load_metrics_from_file():
    if os.path.exists(METRICS_FILE):
        df = pd.read_csv(METRICS_FILE)
        return (
            df["epoch"].tolist(),
            df["train_loss"].tolist(),
            df["val_loss"].tolist(),
            df["train_dice"].tolist(),
            df["val_dice"].tolist(),
            df["train_pc"].tolist(),
            df["val_pc"].tolist(),
        )
    else:
        return [], [], [], [], [], [], []


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward pass
        with torch.amp.autocast(DEVICE):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update loop
        loop.set_postfix(loss=loss.item())
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


# Function for the validation loop
def val_fn(loader, model, loss_fn, scaler):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device=DEVICE)
            targets = targets.float().unsqueeze(1).to(device=DEVICE)
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    model.train()
    return avg_loss


def main():
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5
    )
    scaler = torch.amp.GradScaler(DEVICE)

    train_mean, train_std, val_mean, val_std = get_transforms_value(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        train_transforms=transforms.Compose([transforms.ToTensor()]),
        val_transforms=transforms.Compose([transforms.ToTensor()]),
    )

    print(train_mean, train_std, val_mean, val_std)

    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor(train_mean), torch.Tensor(train_std)),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor(val_mean), torch.Tensor(val_std)),
        ]
    )

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transforms,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    start_epoch = 0
    epoch_history = []
    train_loss_history = []
    val_loss_history = []
    train_dice_history = []
    val_dice_history = []
    train_pc_history = []
    val_pc_history = []

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
        (
            epoch_history,
            train_loss_history,
            val_loss_history,
            train_dice_history,
            val_dice_history,
            train_pc_history,
            val_pc_history,
        ) = load_metrics_from_file()
        start_epoch = epoch_history[-1]

    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        print(f"The {epoch + 1} epoch: start training!")
        epoch_history.append(epoch + 1)

        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        train_loss_history.append(train_loss)

        print("The model predicts on training set:")
        train_pc, train_dice = check_accuracy(train_loader, model, device=DEVICE)
        print(f"Train Loss: {train_loss:.4f}")
        train_dice_history.append(train_dice)
        train_pc_history.append(train_pc)

        print("The model predicts on validating set:")
        val_loss = val_fn(val_loader, model, loss_fn, scaler)
        val_loss_history.append(val_loss)
        val_pc, val_dice = check_accuracy(val_loader, model, device=DEVICE)
        print(f"Val Loss: {val_loss:.4f}")
        val_dice_history.append(val_dice)
        val_pc_history.append(val_pc)

        save_metrics_to_file(
            epoch_history,
            train_loss_history,
            val_loss_history,
            train_dice_history,
            val_dice_history,
            train_pc_history,
            val_pc_history,
        )
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    # # Plot and save training metrics
    # plt.plot(epoch_history, train_loss_history, "r-o", label="Train Loss")
    # plt.plot(epoch_history, val_loss_history, "b-o", label="Val Loss")
    # plt.title("Loss Curve")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("loss_curve.png", dpi=1024)
    # plt.cla()

    # plt.plot(epoch_history, train_dice_history, "r-o", label="Train Dice")
    # plt.plot(epoch_history, val_dice_history, "b-o", label="Val Dice")
    # plt.title("Dice Curve")
    # plt.xlabel("Epoch")
    # plt.ylabel("Dice")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("dice_curve.png", dpi=1024)
    # plt.cla()

    # plt.plot(epoch_history, train_pc_history, "r-o", label="Train Pixel Accuracy")
    # plt.plot(epoch_history, val_pc_history, "b-o", label="Val Pixel Accuracy")
    # plt.title("Pixel Accuracy Curve")
    # plt.xlabel("Epoch")
    # plt.ylabel("Pixel Accuracy")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("pixel_accuracy_curve.png", dpi=1024)
    # plt.cla()


if __name__ == "__main__":
    main()
