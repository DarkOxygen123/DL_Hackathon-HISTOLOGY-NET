# !/usr/bin/env python3
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from preTrain_UNET import UNet11, UNet16

import torch.nn as nn


def preprocess_images(data_folder, patch_size=32, num_patches=4):
    """
    Preprocess images by rotating and saving them in separate directories.

    Args:
        data_folder (str): Path to the folder containing the images.
        patch_size (int, optional): Size of the patches. Defaults to 32.
        num_patches (int, optional): Number of patches to generate. Defaults to 4.
    """
    os.makedirs("Patch_rotated", exist_ok=True)
    os.makedirs("original_rotated", exist_ok=True)

    for filename in tqdm(os.listdir(data_folder), desc="Processing images"):
        img = Image.open(os.path.join(data_folder, filename))

        for angle in range(0, 360, 36):
            img_rotated = img.rotate(angle)

            mask_indices = np.random.randint(
                0, image.shape[0] - patch_size, (num_patches, 2))

            modified_image = np.copy(img_rotated)

            for i in range(num_patches):
                patch_start_x, patch_start_y = mask_indices[i]
                modified_image[patch_start_x:patch_start_x + patch_size,
                               patch_start_y:patch_start_y + patch_size, :] = np.random.randint(0, 256, (patch_size, patch_size, 3))

            modified_image.save(f"Patch_rotated/{filename}_{angle}.jpeg")
            img_rotated.save(f"original_rotated/{filename}_{angle}.jpeg")


class OriginalPatchDataset(Dataset):
    def __init__(self, original_folder, Patch_folder, transform=None):
        """
        Dataset class for loading original and patch images.

        Args:
            original_folder (str): Path to the folder containing the original images.
            Patch_folder (str): Path to the folder containing the patch images.
            transform (callable, optional): Optional transform to be applied to the images. Defaults to None.
        """
        self.original_folder = original_folder
        self.Patch_folder = Patch_folder
        self.transform = transform
        self.original_images = os.listdir(original_folder)
        self.Patch_images = os.listdir(Patch_folder)
        self.original_images.sort()
        self.Patch_images.sort()

    def __len__(self):
        return len(self.original_images)

    def __getitem__(self, idx):
        original_img_path = os.path.join(
            self.original_folder, self.original_images[idx])
        Patch_img_path = os.path.join(
            self.Patch_folder, self.Patch_images[idx])

        original_img = Image.open(original_img_path).convert('RGB')
        Patch_img = Image.open(Patch_img_path).convert('RGB')

        if self.transform:
            original_img = self.transform(original_img)
            Patch_img = self.transform(Patch_img)

        label = 1   # Original image is labeled as 1, Patch image is labeled as 0

        return original_img, Patch_img, label


def create_data_loaders(train_dataset, val_dataset, batch_size=32):
    """
    Create data loaders for training and validation datasets.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        batch_size (int, optional): Batch size. Defaults to 32.

    Returns:
        DataLoader: Training data loader.
        DataLoader: Validation data loader.
    """
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """
    Train the model.

    Args:
        model (nn.Module): Model to be trained.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        num_epochs (int, optional): Number of epochs. Defaults to 10.
    """
    for epoch in range(num_epochs):
        model.train()
        for i, (original_images, Patch_images, _) in enumerate(train_loader):
            outputs = model(Patch_images)
            loss = criterion(outputs, original_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for original_images, Patch_images, _ in val_loader:
                outputs = model(Patch_images)
                loss = criterion(outputs, original_images)
                total_loss += loss.item()

            avg_loss = total_loss / len(val_loader)
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_loss}')

        torch.save(model.state_dict(),
                   f'PatchMask_Unet11/PatchMask_Unet11{epoch}.pth')


def main():
    """
    Main function for training the model.
    """
    # Define the data folder containing the unlabelled images
    data_folder = "/path/to/your/data/folder"

    if not os.path.exists("Patch_rotated"):
        preprocess_images(data_folder)

    # Save pth checkpoint files in a new directory
    # (Change Name of the directory if you want to save the checkpoint files in a different directory and for different UNet configurations)
    if not os.path.exists("PatchMask_Unet11"):
        os.makedirs("PatchMask_Unet11", exist_ok=True)

    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # below folders are created from "preprocess_images" function (Change name if needed in said function)
    original_train_folder = "original_rotated"
    Patch_train_folder = "Patch_rotated"

    dataset = OriginalPatchDataset(
        original_train_folder, Patch_train_folder, transform=data_transform)

    # 80% of the data is used for training
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset)

    model = UNet11(pretrained=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, val_loader, criterion, optimizer)


if __name__ == "__main__":
    main()
