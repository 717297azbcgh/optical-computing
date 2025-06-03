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

# hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 20
NUM_WORKERS = 4
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "./brain/train/images/"
TRAIN_MASK_DIR = "./brain/train/masks/"
VAL_IMG_DIR = "./brain/val/images/"
VAL_MASK_DIR = "./brain/val/masks/"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        # print(data.shape)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        # print(targets.shape)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    return loss.item()


def main():

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
            # transforms.RandomResizedCrop(64),
            # transforms.RandomHorizontalFlip(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor(val_mean), torch.Tensor(val_std)),
        ]
    )
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()
    epoch_history = []
    loss_history = []
    pc_history = []
    dice_history = []
    for epoch in range(NUM_EPOCHS):
        epoch_item = epoch + 1
        print("The ", epoch_item, " epoch:start training!")
        epoch_history.append(epoch_item)
        loss_item = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        loss_history.append(loss_item)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        print("The model predicts on training set:")
        pc_trained, dice_trained = check_accuracy(train_loader, model, device=DEVICE)
        print("The model predicts on validating set:")
        # check accuracy
        pc_item, dice_item = check_accuracy(val_loader, model, device=DEVICE)
        pc_item = pc_item.to(device="cpu")
        dice_item = dice_item.to(device="cpu")
        pc_history.append(pc_item)
        dice_history.append(dice_item)

        """
        #print some examples to a folder
        save_predictions_as_imgs(
            val_loader,model,folder="benign_val_saved_images/",device=DEVICE
        )
        """

    # save the model training process as images
    plt.plot(epoch_history, loss_history, "r-o")
    plt.title("The loss-curve for the training process of benign-segmentation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True)
    plt.savefig("benign_loss_curve", dpi=1024)
    plt.cla()

    plt.plot(epoch_history, pc_history, "r-o")
    plt.title(
        "The pixel-accuracy-curve for the training process of benign-segmentation"
    )
    plt.xlabel("epoch")
    plt.ylabel("pixel-accuracy")
    plt.grid(True)
    plt.savefig("benign_pixel_accuracy_curve", dpi=1024)
    plt.cla()

    plt.plot(epoch_history, dice_history, "r-o")
    plt.title("The dice-curve for the training process of benign-segmentation")
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.grid(True)
    plt.savefig("benign_dice_curve", dpi=1024)


if __name__ == "__main__":
    main()
