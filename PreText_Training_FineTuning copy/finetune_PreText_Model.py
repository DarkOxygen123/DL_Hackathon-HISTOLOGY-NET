
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
import torch
import argparse

import torch.nn as nn
import torch.optim as optim
import ternausnet.models


import torch.nn as nn
import ternausnet.models
from Imports.loss import DiceLoss


class SegmentationModel:
    """
    A class representing a segmentation model.

    Args:
        model_path (str): The path to the pre-trained model.

    Attributes:
        device (torch.device): The device to be used for computation.
        model (torch.nn.Module): The segmentation model.

    Methods:
        load_model(model_path): Loads the pre-trained model.
        replace_final_layer(num_output_features): Replaces the final layer of the model.
        replace_decoder_blocks(num_filters): Replaces the decoder blocks of the model.
        conv3x3(in_: int, out: int) -> nn.Module: Creates a 3x3 convolutional layer.
        conv_relu(in_: int, out: int) -> nn.Module: Creates a convolutional layer followed by ReLU activation.
        decoder_block(in_channels: int, middle_channels: int, out_channels: int) -> nn.Module: Creates a decoder block.

    """

    def __init__(self, model_path):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """
        Loads the pre-trained model.

        Args:
            model_path (str): The path to the pre-trained model.

        Returns:
            torch.nn.Module: The loaded pre-trained model.

        """
        model = ternausnet.models.UNet11(pretrained=True)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def replace_final_layer(self, num_output_features):
        """
        Replaces the final layer of the model.

        Args:
            num_output_features (int): The number of output features for the final layer.

        """
        num_features = self.model.final.in_channels
        self.model.final = nn.Conv2d(
            num_features, num_output_features, kernel_size=1)

    def replace_decoder_blocks(self, num_filters):
        """
        Replaces the decoder blocks of the model.

        Args:
            num_filters (int): The number of filters for the decoder blocks.

        """
        self.model.center = self.decoder_block(
            num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.model.dec5 = self.decoder_block(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.model.dec4 = self.decoder_block(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.model.dec3 = self.decoder_block(
            num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.model.dec2 = self.decoder_block(
            num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.model.dec1 = self.conv_relu(num_filters * (2 + 1), num_filters)

    @staticmethod
    def conv3x3(in_: int, out: int) -> nn.Module:
        """
        Creates a 3x3 convolutional layer.

        Args:
            in_ (int): The number of input channels.
            out (int): The number of output channels.

        Returns:
            torch.nn.Module: The 3x3 convolutional layer.

        """
        return nn.Conv2d(in_, out, 3, padding=1)

    @staticmethod
    def conv_relu(in_: int, out: int) -> nn.Module:
        """
        Creates a convolutional layer followed by ReLU activation.

        Args:
            in_ (int): The number of input channels.
            out (int): The number of output channels.

        Returns:
            torch.nn.Module: The convolutional layer followed by ReLU activation.

        """
        return nn.Sequential(
            SegmentationModel.conv3x3(in_, out),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def decoder_block(in_channels: int, middle_channels: int, out_channels: int) -> nn.Module:
        """
        Creates a decoder block.

        Args:
            in_channels (int): The number of input channels.
            middle_channels (int): The number of middle channels.
            out_channels (int): The number of output channels.

        Returns:
            torch.nn.Module: The decoder block.

        """
        return nn.Sequential(
            SegmentationModel.conv_relu(in_channels, middle_channels),
            nn.ConvTranspose2d(
                middle_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
        )


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.img_names[idx])

        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = Image.merge("RGB", (image, image, image))
        mask = Image.open(mask_path)

        image = self.transform(image)
        mask = self.mask_transform(mask)
        mask = transforms.Lambda(lambda x: (x > 0.5).float())(mask)

        return image, mask


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            progress_bar = tqdm(self.train_dataloader, desc="Train")
            for images, masks in progress_bar:
                images = images.to(self.model.device)
                masks = masks.to(self.model.device)
                outputs = self.model(images)
                outputs = torch.sigmoid(outputs)
                loss = self.criterion(outputs, masks)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                progress_bar.set_postfix(
                    {'train_loss': loss.item()/len(images)})
            print(f'Epoch: {epoch}, Train Loss: {
                  train_loss/len(self.train_dataloader)}')

            # Validation
            self.model.eval()
            val_loss = 0
            progress_bar = tqdm(self.val_dataloader, desc="Validate")
            with torch.no_grad():
                for images, masks in progress_bar:
                    images = images.to(self.model.device)
                    masks = masks.to(self.model.device)
                    outputs = self.model(images)
                    outputs = torch.sigmoid(outputs)
                    loss = self.criterion(outputs, masks)
                    val_loss += loss.item()
                    progress_bar.set_postfix(
                        {'val_loss': loss.item()/len(images)})
            print(f'Epoch: {epoch}, Val Loss: {
                  val_loss/len(self.val_dataloader)}')

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            }, f'model_{epoch}.pth')
            print("Model Saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FineTune PreText Model.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pre-trained model.")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Directory containing the images.")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="Directory containing the masks.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for training.")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs for training.")
    args = parser.parse_args()

    # Load the pre-trained model
    model = SegmentationModel(args.model_path)

    # Replace the final layer and decoder blocks of the model
    model.replace_final_layer(1)
    model.replace_decoder_blocks(32)

    # Define the transformations for the dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mask_transform = transforms.Compose([transforms.ToTensor()])

    # Create the dataset
    dataset = SegmentationDataset(
        args.img_dir, args.mask_dir, transform=transform, mask_transform=mask_transform)

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = train_test_split(
        dataset, test_size=0.2, random_state=42)

    # Create the dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False)

    # Define the criterion and optimizer
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create the trainer
    trainer = Trainer(model, train_dataloader, val_dataloader,
                      criterion, optimizer, args.num_epochs)

    # Train the model
    trainer.train()
