#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


def main():
    """
    Entry point of the program.
    """
    # Define the data folder containing the unlabelled images
    data_folder = "/path/to/your/data/folder"

    # Check if the directories for rotated and gray images exist
    if not os.path.exists("gray_rotated"):
        preprocess_images(data_folder)

    # Save pth checkpoint files in a new directory (Change Name of the directory if you want to save the checkpoint files in a different directory and for different UNet configurations)
    if not os.path.exists("colourize_Unet11"):
        os.makedirs("colourize_Unet11", exist_ok=True)

    # Define data transforms
    data_transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
    ])

    # Define paths to image folders
    original_train_folder = "original_rotated"
    gray_train_folder = "gray_rotated"

    # Create dataset
    dataset = OriginalGrayDataset(
        original_train_folder, gray_train_folder, transform=data_transform)

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset)

    # Instantiate the model
    model = UNet11(pretrained=True)
    # model = UNet16(pretrained=True)  ##### Run with UNet11 model once to get a checkpoint file for this (The final Ensemble model uses UNet16 model and UNet11 model for better results)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer)


class OriginalGrayDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading original and gray images.
    """

    def __init__(self, original_folder, gray_folder, transform=None):
        """
        Initialize the dataset.

        Args:
            original_folder (str): Path to the folder containing original images.
            gray_folder (str): Path to the folder containing gray images.
            transform (callable, optional): Optional transform to be applied to the images.
        """
        self.original_folder = original_folder
        self.gray_folder = gray_folder
        self.transform = transform

        # Load image paths
        self.original_images = os.listdir(original_folder)
        self.gray_images = os.listdir(gray_folder)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.original_images)

    def __getitem__(self, idx):
        """
        Returns the original and gray image at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the original and gray images.
        """
        original_image_path = os.path.join(
            self.original_folder, self.original_images[idx])
        gray_image_path = os.path.join(self.gray_folder, self.gray_images[idx])

        original_image = Image.open(original_image_path)
        gray_image = Image.open(gray_image_path)

        if self.transform:
            original_image = self.transform(original_image)
            gray_image = self.transform(gray_image)

        return original_image, gray_image


def train_model(model, train_loader, val_loader, criterion, optimizer):
    """
    Trains the model

    Args:
        model (nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
    """
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Print epoch statistics
        print(f"Epoch {
              epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
