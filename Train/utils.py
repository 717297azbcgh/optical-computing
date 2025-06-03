import torch
import torchvision
from dataset import GallBladderDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    print("=> Loaded checkpoint successfully")


def get_mean_std(dataset):
    print(len(dataset))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )
    mean = torch.zeros(1)
    std = torch.zeros(1)
    for x, _ in loader:  # 计算loader中所有数据的mean和atd的累积
        mean += x.detach().mean()
        std += x.detach().std()
    mean = torch.div(mean, len(dataset))  # 得到整体数据集mean的平均
    std = torch.div(std, len(dataset))
    return list(mean.numpy()), list(std.numpy())  # 返回mean和std的list


def get_transforms_value(
    train_img_dir,
    train_mask_dir,
    val_img_dir,
    val_mask_dir,
    train_transforms,
    val_transforms,
):
    train_ds = GallBladderDataset(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        transform=train_transforms,
    )

    train_mean, train_std = get_mean_std(train_ds)

    val_ds = GallBladderDataset(
        image_dir=val_img_dir,
        mask_dir=val_mask_dir,
        transform=val_transforms,
    )

    val_mean, val_std = get_mean_std(val_ds)

    return train_mean, train_std, val_mean, val_std


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = GallBladderDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = GallBladderDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def get_test_loader(
    test_dir,
    test_maskdir,
    batch_size,
    test_transform,
    num_workers=4,
    pin_memory=True,
):
    test_ds = GallBladderDataset(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return test_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    with torch.no_grad():
        loop = tqdm(loader)
        for x, y in loop:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    Pixel_Accuracy = num_correct / num_pixels
    Dice_Coefficient = dice_score / len(loader)
    print(f"Got {num_correct}/{num_pixels} with Pixel Accuracy {Pixel_Accuracy}")
    print(f"Dice Coefficient:{Dice_Coefficient}")
    model.train()
    return Pixel_Accuracy, Dice_Coefficient


# print some predicting images during the training process
def save_predictions_as_imgs(loader, model, folder="val_saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}y_{idx}.png")

    model.train()
