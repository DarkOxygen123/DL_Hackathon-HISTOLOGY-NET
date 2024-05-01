import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import ternausnet.models
from Imports.loss import DiceLoss

model = ternausnet.models.UNet11(pretrained=True)


class SegmentationDataset(Dataset):
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


transform = transforms.Compose([transforms.ToTensor()])
mask_transform = transforms.Compose([transforms.ToTensor()])

if not os.path.exists('models'):
    os.makedirs('models')

# Path to the directory containing images (Augmented Labelled)
img_dir = 'final_images'
# Path to the directory containing masks (Augmented Labelled)
mask_dir = 'final_masks'
dataset = SegmentationDataset(img_dir=img_dir, mask_dir=mask_dir,
                              transform=transform, mask_transform=mask_transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = DiceLoss()
optimizer = Adam(model.parameters(), lr=0.001)

num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    loss_train_total = 0.0
    progress_bar = tqdm(dataloader, desc='Train', total=len(dataloader))
    for i, batch in enumerate(progress_bar):
        images, masks = batch
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        outputs = torch.sigmoid(outputs)

        if (i % 100 == 0):
            image3 = images[0].cpu().squeeze().numpy()
            cv2.imwrite(f"results51/{epoch+15}_{i}_original.jpeg",
                        (image3 * 255).astype('uint8').transpose((1, 2, 0)))
            image3 = outputs[0].cpu().squeeze().detach().numpy()
            cv2.imwrite(
                f"results51/{epoch+15}_{i}_output.jpeg", (image3 * 255).astype('uint8'))
            image3 = masks[0].cpu().squeeze().numpy()
            cv2.imwrite(f"results51/{epoch+15}_{i}_mask.jpeg",
                        (image3 * 255).astype('uint8'))

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        loss_train_total += loss.item()
        progress_bar.set_postfix(
            {'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

    loss_train_avg = loss_train_total / len(dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss_train_avg}')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'models/model_{epoch}.pth')
